import yaml
import subprocess
import time
import click
import pathlib
import os
import sys
import json
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
from contextlib import contextmanager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('queue_manager.log')
    ]
)


@dataclass
class TaskStatus:
    filename: str
    start_time: str
    pid: int
    hostname: str


class QueueError(Exception):
    """Base class for queue-related errors"""
    pass


class QueueManager:
    def __init__(self, queue_file: pathlib.Path):
        self.queue_file = queue_file
        self.status_file = queue_file.parent / '.queue_status.json'
        self.hostname = os.uname().nodename

    def validate_queue_file(self) -> bool:
        """Validate queue file structure"""
        try:
            data = self.read_queue()
            required_keys = {'on_queue', 'doing', 'done'}
            if not all(key in data for key in required_keys):
                raise QueueError(f"Missing required keys in queue file. Need: {required_keys}")
            if not isinstance(data['on_queue'], list):
                raise QueueError("'on_queue' must be a list")
            if not isinstance(data.get('done', []), list):
                raise QueueError("'done' must be a list")
            return True
        except yaml.YAMLError as e:
            raise QueueError(f"Invalid YAML format: {e}")

    @contextmanager
    def file_lock(self):
        """Simple file locking mechanism to prevent race conditions"""
        lock_file = self.queue_file.parent / '.queue.lock'
        try:
            while lock_file.exists():
                time.sleep(0.1)
            lock_file.touch()
            yield
        finally:
            if lock_file.exists():
                lock_file.unlink()

    def read_queue(self):
        try:
            with self.file_lock():
                with open(self.queue_file, 'r') as f:
                    return yaml.safe_load(f) or {'on_queue': [], 'doing': None, 'done': []}
        except yaml.YAMLError as e:
            logging.error(f"Error reading queue file: {e}")
            raise

    def write_queue(self, queue_data):
        try:
            with self.file_lock():
                with open(self.queue_file, 'w') as f:
                    yaml.dump(queue_data, f, default_flow_style=False)
        except Exception as e:
            logging.error(f"Error writing queue file: {e}")
            raise

    def save_task_status(self, task: str, pid: int):
        status = TaskStatus(
            filename=task,
            start_time=datetime.now().isoformat(),
            pid=pid,
            hostname=self.hostname
        )

        try:
            with open(self.status_file, 'w') as f:
                json.dump(asdict(status), f)
        except Exception as e:
            logging.error(f"Error saving task status: {e}")
            raise

    def clear_task_status(self):
        try:
            if self.status_file.exists():
                self.status_file.unlink()
        except Exception as e:
            logging.error(f"Error clearing task status: {e}")

    def check_interrupted_task(self) -> Optional[str]:
        """Check if there was an interrupted task from a shutdown"""
        if not self.status_file.exists():
            return None

        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)

            if status['hostname'] != self.hostname:
                logging.warning(f"Found task running on different machine: {status['hostname']}")
                return None

            try:
                os.kill(status['pid'], 0)
                logging.warning(f"Process {status['pid']} still running")
                return None
            except OSError:
                logging.info(f"Found interrupted task: {status['filename']}")
                return status['filename']

        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Corrupt status file: {e}")
            self.clear_task_status()
            return None

    def get_next_task(self) -> Optional[str]:
        queue_data = self.read_queue()

        interrupted_task = self.check_interrupted_task()
        if interrupted_task:
            if queue_data.get('doing') != interrupted_task:
                logging.warning("Queue file and status file mismatch. Using status file.")
                queue_data['doing'] = interrupted_task
                self.write_queue(queue_data)
            return interrupted_task

        if queue_data.get('doing'):
            return queue_data['doing']

        if queue_data['on_queue']:
            task = queue_data['on_queue'].pop(0)
            queue_data['doing'] = task
            self.write_queue(queue_data)
            return task

        return None

    def mark_as_done(self, task: str):
        queue_data = self.read_queue()
        if queue_data.get('doing') == task:
            queue_data['done'] = queue_data.get('done', []) + [task]
            queue_data['doing'] = None
            self.write_queue(queue_data)
            self.clear_task_status()

    def run_training(self, config_file: str, checkpoint_path: str, load_on_fit: bool) -> bool:
        cmd = [
            "python",
            "train_pore.py",
            f"--cfgpath={config_file}",
            f"--checkpoint_path={checkpoint_path}",
        ]
        if load_on_fit:
            cmd.append("--load_on_fit")

        logging.info(f"Running command: {' '.join(cmd)}")

        process = None
        try:
            process = subprocess.Popen(cmd)
            self.save_task_status(config_file, process.pid)

            # Monitor process with timeout
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=10)  # Check every 10 seconds
                    if process.returncode is not None:
                        if process.returncode != 0:
                            logging.error(f"Process failed with return code {process.returncode}")
                            logging.error(f"Stderr: {stderr}")
                            return False
                        return True
                except subprocess.TimeoutExpired:
                    # Process still running, check if we want to continue
                    continue

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)  # Give it 5 seconds to terminate
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if it doesn't terminate
            return False
        except Exception as e:
            logging.error(f"Unexpected error during training: {e}")
            if process:
                process.terminate()
            return False


@click.command()
@click.option('--queue-file', type=click.Path(exists=True), required=True,
              help='Path to the YAML queue file')
@click.option('--checkpoint-path', type=str, default='last',
              help='Checkpoint path to use for training')
@click.option('--force-clear-status', is_flag=True, default=False,
              help='Force clear any existing status file before starting')
def main(queue_file: str, checkpoint_path: str, force_clear_status: bool):
    load_on_fit = True
    """Manage a queue of training tasks with shutdown protection."""
    try:
        queue_file = pathlib.Path(queue_file)
        manager = QueueManager(queue_file)

        # Validate queue file before starting
        manager.validate_queue_file()

        if force_clear_status:
            manager.clear_task_status()

        while True:
            task = manager.get_next_task()
            if not task:
                logging.info("No more tasks in queue")
                break

            logging.info(f"Processing task: {task}")
            success = manager.run_training(task, checkpoint_path, load_on_fit)

            if success:
                logging.info(f"Task completed successfully: {task}")
                manager.mark_as_done(task)
            else:
                logging.warning(f"Task failed or was interrupted: {task}")
                break

    except QueueError as e:
        logging.error(f"Queue error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()
