# Fetch all branches
git fetch --all

# Loop through each branch and remove .ipynb files from tracking without deleting them
for branch in $(git branch -r | grep -v '\->'); do
    branch_name=${branch#origin/}
    git checkout $branch_name

    # Remove .ipynb files in the specified directory from tracking without deleting them
    find notebooks/exploratory -name '*.ipynb' -not -name 'GITSHARE*.ipynb' -exec git rm --cached {} \;

    # Commit the changes locally
    git commit -m "Stop tracking .ipynb files in notebooks/exploratory/ directory"

    # Optional: Push the changes if needed
    git push origin $branch_name
done

# Clean up local branches
git checkout main