#!/bin/bash

git -C ../src ls-files --exclude-standard -oi --directory >> ../.rsync-exclude

echo "Local to Remote"
sshpass -p "vvlfl167" rsync --exclude=__pycache__ --exclude=.git --exclude-from="../.rsync-exclude" -rthP --update /mnt/f/Users/Erwin/Documents/GitHub/AdvancedComputing/src/ group27@ce-tesla.et.tudelft.nl:~/projects/

if [[ $? -eq 0 ]]; then
    echo "Remote to Local"
    #--exclude=.git --exclude-from="../.rsync-exclude" 
    sshpass -p "vvlfl167" rsync -rthP --update group27@ce-tesla.et.tudelft.nl:~/projects/ /mnt/f/Users/Erwin/Documents/GitHub/AdvancedComputing/src/
fi
