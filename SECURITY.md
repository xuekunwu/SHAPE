# Security Notice

## API Key Exposure

A previous commit in this repository's history contained an exposed OpenAI API key. 

**If you are using this repository:**

1. **IMMEDIATELY revoke the exposed API key** in your OpenAI account:
   - Go to https://platform.openai.com/api-keys
   - Revoke the key that starts with `sk-proj-rYKiIBVBNLOjATsphNg5dgiEB...`
   - Generate a new API key

2. **Do not use the exposed key** - it has been compromised

3. **For future commits**: Never commit API keys, tokens, or secrets to git. Use environment variables or secure secret management.

## Removing Secrets from Git History

To completely remove the secret from git history, you can use:

```bash
# Using git filter-repo (recommended)
git filter-repo --path README.md --invert-paths
# Then re-add README.md with clean content

# Or using BFG Repo-Cleaner
bfg --replace-text passwords.txt
```

**Note**: Rewriting git history requires force-pushing and will affect all collaborators.

