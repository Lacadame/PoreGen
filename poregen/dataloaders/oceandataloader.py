import pathlib
import os

import torch
import torchvision.transforms.v2 as transforms
import lightning
import numpy as np
import netCDF4

import datetime


class ContiguousCurrentsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 directory,
                 subwindow=None,
                 D=3,
                 transform=None,
                 skip=1):
        self.directory = directory
        self.subwindow = subwindow
        self.D = D
        if skip == 0:
            skip += 1
        self.skip = skip
        self.files_2du = sorted([f
                                 for f in os.listdir(directory)
                                 if '2du' in f])[::skip]
        self.files_fsd = sorted([f
                                 for f in os.listdir(directory)
                                 if 'fsd' in f])[::skip]
        self.paths_2du = {f.split('_')[1] + '_' + f.split('_')[2]:
                          os.path.join(directory, f) for f in self.files_2du}
        self.paths_fsd = {f.split('_')[1] + '_' + f.split('_')[2]:
                          os.path.join(directory, f) for f in self.files_fsd}
        self.paths_bat = directory/'batimetria_cortada_interpolada_LSE36.nc'
        self.day_hours, self.hours_since_year_start = self.extract_hours_by_day(self.files_2du)
        self.transform = transform
        self.valid_indices = self.calculate_valid_indices()

    def extract_hours_by_day(self, strings):
        day_to_hours = {}
        for s in strings:
            parts = s.split('_')
            day, hour = parts[1], parts[2]
            if day not in day_to_hours:
                day_to_hours[day] = []
            day_to_hours[day].append(hour)
        day_hours = sorted((day, hour)
                           for day, hours in day_to_hours.items()
                           for hour in hours)
        hours_since_start = {f'{day}_{hour}':
                             24 * int(day) + int(hour)
                             for day, hours in day_to_hours.items()
                             for hour in hours}
        # print('day_hours', day_hours)
        # print('\nhours_since_start',  hours_since_start)
        return day_hours, hours_since_start

    def calculate_valid_indices(self):
        valid_indices = []

        for i in range(len(self.day_hours) - self.D + 1):

            start_day, start_hour = self.day_hours[i]

            start_hour_index = self.hours_since_year_start[
                f'{start_day}_{start_hour}']

            expected_end_hour_index = start_hour_index + (self.D-1)*self.skip

            end_day, end_hour = self.day_hours[i + self.D - 1]

            if (self.hours_since_year_start.get(
                    f'{end_day}_{end_hour}', -1
                    ) == expected_end_hour_index):

                # To make sure we dont have diferent hours from diferent days

                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx, subwindow=None):
        if subwindow is None:
            subwindow = self.subwindow[np.random.choice(len(self.subwindow))]
        sequence_data = []
        actual_idx = self.valid_indices[idx]

        for offset in range(self.D):
            day, hour = self.day_hours[actual_idx + offset]
            filepath_2du = self.paths_2du[f'{day}_{hour}']
            filepath_fsd = self.paths_fsd[f'{day}_{hour}']
            filepath_bat = self.paths_bat
            with (netCDF4.Dataset(filepath_2du, "r") as nc_dataset_2du,
                  netCDF4.Dataset(filepath_fsd, "r") as nc_dataset_fsd,
                  netCDF4.Dataset(filepath_bat, "r") as nc_dataset_bat):

                latitudes = nc_dataset_2du.variables['Latitude'][:]
                longitudes = nc_dataset_2du.variables['Longitude'][:]


                lat_idx, lon_idx = self.get_indices_from_proportion(latitudes,
                                                                    longitudes,
                                                                    subwindow)

                u_velocity = nc_dataset_2du.variables['u_velocity'][
                    0, 0, lat_idx[0]:lat_idx[1], lon_idx[0]:lon_idx[1]]

                v_velocity = nc_dataset_2du.variables['v_velocity'][
                    0, 0, lat_idx[0]:lat_idx[1], lon_idx[0]:lon_idx[1]]

                ssh = nc_dataset_fsd.variables['ssh'][
                    0, lat_idx[0]:lat_idx[1], lon_idx[0]:lon_idx[1]]

                latitudes = latitudes[lat_idx[0]:lat_idx[1]]
                longitudes = longitudes[lon_idx[0]:lon_idx[1]]

                bat = nc_dataset_bat.variables['batim'][
                    lat_idx[0]:lat_idx[1], lon_idx[0]:lon_idx[1]]
               
                mask = np.ma.getmask(u_velocity)
                if len(mask.shape) == 0:
                    mask = np.zeros_like(u_velocity, dtype=bool)
                sequence_data.append((u_velocity, v_velocity, ssh,
                                      mask, latitudes, longitudes, bat))

        tensors = [self.prepare_tensors(*data) for data in sequence_data]

        tensors = list(zip(*tensors))
        tensors = [torch.stack(t) for t in tensors]
        sizes = [tensor.shape[1] for tensor in tensors]
        tensors = torch.cat(tensors, axis=1)

        # Apply transform
        if self.transform:
            tensors = self.transform(tensors)

        # return to the original dimensions
        result_tensors = torch.split(tensors, sizes, dim=1)

        return result_tensors, (day, hour)
        # Returning the last day and hour of the sequence for reference

    def get_indices_from_proportion(self, latitudes, longitudes, subwindow):
        if subwindow:
            lat_range = [int(subwindow[0][0] * len(latitudes)),
                         int(subwindow[0][1] * len(latitudes))]
            lon_range = [int(subwindow[1][0] * len(longitudes)),
                         int(subwindow[1][1] * len(longitudes))]
            return lat_range, lon_range
        return [0, len(latitudes)], [0, len(longitudes)]

    def prepare_tensors(self, u_velocity, v_velocity,
                        ssh, mask, latitudes, longitudes, bat):
        u_tensor = torch.tensor(u_velocity.filled(0.0), dtype=torch.float32)
        v_tensor = torch.tensor(v_velocity.filled(0.0), dtype=torch.float32)
        ssh_tensor = torch.tensor(ssh.filled(0.0), dtype=torch.float32)
        combined_tensor = torch.stack([u_tensor, v_tensor, ssh_tensor])
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        latitudes = torch.tensor(latitudes, dtype=torch.float32)

        longitudes = torch.tensor(longitudes, dtype=torch.float32)
        latitudes, longitudes = torch.meshgrid(latitudes, longitudes)

        latitudes = latitudes.unsqueeze(0)
        longitudes = longitudes.unsqueeze(0)

        bat_tensor = torch.tensor(bat, dtype=torch.float32).unsqueeze(0)
        # Remove any possible nans, converting to 0.0
        combined_tensor = torch.nan_to_num(combined_tensor, nan=0.0)
        mask_tensor = torch.nan_to_num(mask_tensor, nan=1.0)
        bat_tensor = torch.nan_to_num(bat_tensor, nan=1.0)

        return mask_tensor, combined_tensor, latitudes, longitudes, bat_tensor


class ContiguousCurrentsDatasetAutoregressive(ContiguousCurrentsDataset):
    def __init__(self, directory, subwindow=None, transform=None, D=3, skip=1,
                 ssh_only=False, r_day=False, r_latlon=False, r_bat=False):
        self.r_day = r_day
        self.r_latlon = r_latlon
        self.r_bat = r_bat
        super().__init__(directory, subwindow=subwindow, D=D,
                         transform=transform, skip=skip)
        self.ssh_only = ssh_only

    def __getitem__(self, idx, subwindow=None):
        tensors, (day, hour) = super().__getitem__(idx, subwindow)

        if self.ssh_only:
            x = tensors[1][-1][-1]
            x = x.unsqueeze(0)
            y = tensors[1][0:-1, -1, :, :]

        else:
            x = tensors[1][-1]
            y = tensors[1][0:-1]
            y = y.reshape(y.size(1), y.size(0), y.size(-2), y.size(-1))
            y = y.reshape(-1, y.size(-2), y.size(-1))
        mask = tensors[0][-1]

        y = {'y': y}

        if self.r_day:
            y['dates'] = self.day_of_year_to_date(int(day))

        if self.r_latlon:
            lat = tensors[-3][-1]
            lon = tensors[-2][-1]

            y['latlon'] = torch.tensor([lat[0,63, 0], lon[0,0, 63]])

        if self.r_bat:
            y['bat'] = tensors[-1][0]
        return x, y, mask

    def day_of_year_to_date(self, dia_do_ano):
        ano_bissexto = 2020

        # create inicial data
        data_inicial = datetime.datetime(ano_bissexto, 1, 1)

        data_final = data_inicial + datetime.timedelta(days=dia_do_ano - 1)

        # return mpnth and day
        return torch.tensor([data_final.month, data_final.day])


def split_dataset_from_proportions(dataset, proportions):
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    return torch.utils.data.random_split(dataset, lengths)
