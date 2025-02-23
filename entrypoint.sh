#!/bin/bash

# Set default PUID and PGID if not provided
PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Create a group with PGID if it doesn't exist
if ! getent group appgroup > /dev/null 2>&1; then
    groupadd -g $PGID appgroup
fi

# Create a user with PUID if it doesn't exist
if ! id -u appuser > /dev/null 2>&1; then
    useradd -u $PUID -g $PGID -m -s /bin/bash appuser 2>/dev/null || true
fi

# Ensure proper ownership and permissions for /output
chown -R $PUID:$PGID /output
chown -R $PUID:$PGID /app/assets  # Ensure `appuser` owns the assets folder
chown -R $PUID:$PGID /app/models  # Ensure `appuser` owns the models folder

# Print the build timestamp
echo "Build Timestamp:"
cat /app/build_timestamp.txt

# Switch to the user and execute the CMD
exec gosu $PUID:$PGID "$@"