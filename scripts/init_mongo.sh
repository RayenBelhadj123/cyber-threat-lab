#!/usr/bin/env bash
set -euo pipefail

for attempt in $(seq 1 60); do
  if mongosh --host mongo-1:27017 --quiet --eval 'db.adminCommand({ping:1}).ok' | grep -q 1 \
     && mongosh --host mongo-2:27017 --quiet --eval 'db.adminCommand({ping:1}).ok' | grep -q 1 \
     && mongosh --host mongo-3:27017 --quiet --eval 'db.adminCommand({ping:1}).ok' | grep -q 1; then
    break
  fi
  sleep 2
done

mongosh --host mongo-1:27017 --quiet --eval '
  try {
    rs.status().ok;
  } catch (e) {
    rs.initiate({
      _id: "rs0",
      members: [
        {_id: 0, host: "mongo-1:27017"},
        {_id: 1, host: "mongo-2:27017"},
        {_id: 2, host: "mongo-3:27017"}
      ]
    }).ok;
  }
'

for attempt in $(seq 1 90); do
  if mongosh --host mongo-1:27017 --quiet --eval 'db.hello().isWritablePrimary' | grep -q true; then
    echo 'MongoDB replica set rs0 is ready.'
    exit 0
  fi
  sleep 2
done

echo 'MongoDB replica set did not elect a primary within 180 seconds.' >&2
exit 1
