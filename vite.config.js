import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'node:path';

export default defineConfig({
  plugins: [vue()],
  base: '/',
  publicDir: 'assets',
  build: {
    rollupOptions: {
      input: {
        home: resolve(__dirname, 'index.html'),
        karacter: resolve(__dirname, 'karacter/index.html')
      }
    }
  }
});


