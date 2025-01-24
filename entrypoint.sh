#!/bin/bash

# Set default PUID and PGID if not provided
PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Create a group with PGID if it doesn't exist
if ! getent group $PGID > /dev/null 2>&1; then
    groupadd -g $PGID appgroup
fi

# Create a user with PUID if it doesn't exist
if ! id -u $PUID > /dev/null 2>&1; then
    useradd -u $PUID -g $PGID -m -s /bin/bash appuser
fi

# Ensure proper ownership and permissions for /output
chown -R $PUID:$PGID /output

# Print ownership details for debugging
ls -ld /output
id appuser

# Switch to the user and execute the CMD
exec gosu $PUID:$PGID "$@"