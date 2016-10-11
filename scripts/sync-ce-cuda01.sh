#!/bin/bash

git -C ../src ls-files --exclude-standard -oi --directory >> ../.rsync-exclude

echo "Local to Remote"
sshpass -p "yjzr5pz2" rsync --exclude=__pycache__ --exclude=.git --exclude-from="../.rsync-exclude" -rthP --update /mnt/f/Users/Erwin/Documents/GitHub/AdvancedComputing/src/ group27@ce-cuda01.et.tudelft.nl:~/projects/

if [[ $? -eq 0 ]]; then
    echo "Remote to Local"
    #--exclude=.git --exclude-from="../.rsync-exclude" 
    sshpass -p "yjzr5pz2" rsync -rthP --update group27@ce-cuda01.et.tudelft.nl:~/projects/ /mnt/f/Users/Erwin/Documents/GitHub/AdvancedComputing/src/
fi
