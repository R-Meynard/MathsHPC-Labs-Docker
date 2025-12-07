# Devcontainer profile for container
# Export PATH additions and any environment tweaks here.
export PATH="\$PATH:/usr/local/bin"

# Source any snippets placed in /root/.profile.d inside the container
if [ -d "/root/.profile.d" ]; then
  for f in /root/.profile.d/*; do
    [ -r "\$f" ] && . "\$f"
  done
fi
