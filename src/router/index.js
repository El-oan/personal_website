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
      meta: { 
        title: 'Portfolio',
        favicon: '/favicon.jpeg'
      },
    },
    {
      path: '/karacter',
      name: 'karacter',
      component: KaracterView,
      alias: ['/karacter/'],
      meta: { 
        title: 'Karacter',
        favicon: '/KaracterLogo.png'
      },
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

router.afterEach((to) => {
  document.title = to.meta.title || 'Portfolio';
  
  const link = document.querySelector("link[rel~='icon']");
  if (link) {
    link.href = to.meta.favicon || '/favicon.png';
  }
});

export default router;

