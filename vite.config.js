import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'node:path';

export default defineConfig({
  plugins: [vue()],
  base: '/',
  publicDir: 'assets', 
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
});
