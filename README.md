# Personal Website

**Status:** Live 🟢  
**Website:** [eloantourtelier.com](https://eloantourtelier.com)

This repository contains the source code for my personal website (Vue 3 + Vite).

## Development

```bash
npm install
npm run dev
```

## Deployment

This site is hosted on GitHub Pages. To deploy a new version:

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Update website"
   git push
   ```

2. **Build and Deploy**:
   ```bash
   npm run build
   # Push the 'dist' folder to the 'gh-pages' branch using the gh-pages tool:
   npx gh-pages -d dist
   ```

_(Make sure GitHub Pages is configured to serve from the `gh-pages` branch in your repo settings.)_
