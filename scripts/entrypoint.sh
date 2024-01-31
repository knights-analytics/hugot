#!/bin/bash

#--- entrypoint script to the alchemia stack container
# This script should be called with arguments as follows:
#
# `command arg1 arg2 ...`
#
# where `command` must be one of: alpha, dgraph, or qdrant, and the
# arguments arg1, arg2, ... are passed to the respective executable
# and override the existing default arguments if applicable
#---

user_name=testuser # the container non-privileged user (see dockerfile)

if [[ -n $HOST_UID ]] && [[ $HOST_UID != 1000 ]]; then
    # patching internal $user_name to have the required host user id
    usermod -u "$HOST_UID" $user_name
fi

echo "$1 running with user: $user_name"
chown "$user_name:$user_name" "$1" && chmod +x "$1"
exec runuser -u $user_name -- "$1" "${arguments[@]}"
