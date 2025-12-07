# Devcontainer .bashrc (sourced for interactive shells)
# Load profile if present
[ -f /root/.profile ] && . /root/.profile

# Source snippets if present
if [ -d /root/.bashrc.d ]; then
  for f in /root/.bashrc.d/*; do
    [ -r "$f" ] && . "$f"
  done
fi

# Common customizations
export PATH="$PATH:/usr/local/bin"
alias ll='ls -la'
