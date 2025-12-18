import { createRouter, createWebHistory } from 'vue-router';
import HomeView from '../home/HomeView.vue';
import KaracterView from '../karacter/KaracterView.vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/karacter',
      name: 'karacter',
      component: KaracterView,
      alias: ['/karacter/'],
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/',
    },
  ],
  scrollBehavior(to, from, savedPosition) {
    if (to.hash) {
      return {
        el: to.hash,
        behavior: 'smooth',
      };
    }
    return savedPosition || { top: 0 };
  },
});

export default router;

