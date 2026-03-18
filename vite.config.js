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
        main: resolve(__dirname, 'index.html'),
        'focustrain/index': resolve(__dirname, 'focustrain/index.html'),
        'karacter/index': resolve(__dirname, 'karacter/index.html'),
        'karacter/terms/index': resolve(__dirname, 'karacter/terms/index.html'),
        'notebook/index': resolve(__dirname, 'notebook/index.html'),
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
});
