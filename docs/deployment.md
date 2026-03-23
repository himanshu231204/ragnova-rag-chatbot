# Deployment Guide

## Streamlit Community Cloud

1. Push repository to GitHub.
2. Open https://share.streamlit.io and create a new app.
3. Select repo and branch.
4. Set main file path to `streamlit_app.py`.
5. Add secret in Streamlit settings:

```toml
GROQ_API_KEY = "your_key_here"
```

6. Deploy.

## Instant Startup Strategy (No Manual Rebuild)

To avoid asking users to click "Build/Rebuild Index" on first open:

1. Build index locally once.
2. Commit these files to the repository:
	- `faiss_store/faiss.index`
	- `faiss_store/metadata.pkl`
3. Push to GitHub and redeploy.

With this, app starts ready-to-chat on Streamlit Cloud.

## GitHub Actions

- `ci.yml`: syntax checks
- `test.yml`: pytest suite
- `cd.yml`: optional deployment webhook trigger

## CD Webhook (Optional)

Set `STREAMLIT_DEPLOY_WEBHOOK_URL` as a repository secret if your deployment
platform supports webhook-based redeploy.
