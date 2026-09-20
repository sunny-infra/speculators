if not contains "$HOME/spec_code/speculators/.tools" $PATH
    # Prepending path in case a system-installed binary needs to be overridden
    set -x PATH "$HOME/spec_code/speculators/.tools" $PATH
end
