#!/bin/bash

# Fetch all branches
git fetch --all

# List all remote branches
branches=$(git branch -r | grep -v '\->')

# Loop through each branch
for branch in $branches; do
    # Checkout the branch
    branch_name=${branch#origin/}
    git checkout $branch_name

    # Remove .ipynb files in the specified directory
    find notebooks/exploratory -name '*.ipynb' -not -name 'GITSHARE*.ipynb' -exec git rm --cached {} \;

    # Commit the changes
    git commit -m "Remove .ipynb files from notebooks/exploratory/ directory"

    # Push the changes to the remote repository
    git push origin $branch_name
done

# Clean up local branches
git checkout main

# Use filter-branch to remove .ipynb files from history in the specified directory
git filter-branch --force --index-filter \
  'git rm -r --cached --ignore-unmatch notebooks/exploratory/*.ipynb' \
  --prune-empty --tag-name-filter cat -- --all

# Force push the changes to the remote repository
git push --force --all
git push --force --tags

# Instruct collaborators to clean up their local repositories
echo "Inform collaborators to run the following commands:
# Fetch all branches
git fetch --all

# Checkout each branch and reset it
for branch in \$(git branch -r | grep -v '\->'); do
    branch_name=\${branch#origin/}
    git checkout \$branch_name
    git reset --hard origin/\$branch_name
done

# Clean up local repository
git reflog expire --expire=now --all && git gc --prune=now --aggressive
"
