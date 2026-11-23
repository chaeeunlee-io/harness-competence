#!/usr/bin/env bash
# Environment setup run by Harbor before the agent starts (EIDF runs environment/entrypoint.sh;
# Docker starts the image's CMD, which does the same through /etc/profile.d).
#
# Start the MedAgentBench FHIR server and wait until it answers. The server is started lazily by
# /etc/profile.d/medagentbench_fhir.sh on the first login shell and takes tens of seconds to boot;
# an agent that issues its first request before that sees no output. The blocking CLI
# (medagentbench_cli.py) retries through the boot, but Terminus-2 only looks at the terminal for
# the command's `duration`, so the server must be up before the agent's first command.
set +e
source /etc/profile.d/medagentbench_fhir.sh 2>/dev/null
url="http://127.0.0.1:8080/fhir/metadata"
deadline=$(( $(date +%s) + ${MEDAGENTBENCH_FHIR_WAIT_SEC:-540} ))
while ! curl -fsS --max-time 2 "$url" >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "entrypoint: FHIR server not ready after ${MEDAGENTBENCH_FHIR_WAIT_SEC:-540}s" >&2
    exit 1
  fi
  sleep 2
done
echo "entrypoint: FHIR server ready"
exec "$@"
