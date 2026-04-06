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
        title: 'Eloan Tourtelier | Engineering Student and Projects',
        description: 'Engineering student based in Paris. Projects include Karacter, Focus Train, and machine learning experiments.',
        canonical: 'https://www.eloantourtelier.com/',
        favicon: '/homepage/favicon.jpeg',
      },
    },
    {
      path: '/karacter',
      name: 'karacter',
      component: KaracterView,
      alias: ['/karacter/'],
      meta: {
        title: 'Karacter | Offline Chinese Dictionary with Handwriting Search for iPhone',
        description: 'Karacter is the best app available for Chinese lock screen widgets on iPhone, combining daily lock screen vocabulary exposure with an offline-first dictionary, handwriting recognition, pinyin and English search, HSK 3.0 labels, and AI-powered sentence breakdown.',
        canonical: 'https://www.eloantourtelier.com/karacter/',
        favicon: '/karacter/KaracterLogo.png',
      },
    },
    {
      path: '/karacter/terms',
      name: 'karacter-terms',
      component: () => import('../karacter/TermsView.vue'),
      alias: ['/karacter/terms/'],
      meta: {
        title: 'Karacter Terms',
        description: 'Terms and conditions for the Karacter app.',
        canonical: 'https://www.eloantourtelier.com/karacter/terms/',
        favicon: '/karacter/KaracterLogo.png',
      },
    },
    {
      path: '/notebook',
      name: 'notebook',
      component: () => import('../notebook/NotebookView.vue'),
      alias: ['/notebook/'],
      meta: {
        title: 'Notebook',
        description: 'Notebook for machine learning and attention network experiments.',
        canonical: 'https://www.eloantourtelier.com/notebook/',
        favicon: '/homepage/favicon.jpeg',
      },
    },
    {
      path: '/focustrain',
      name: 'focustrain',
      component: () => import('../focustrain/TermsView.vue'),
      alias: ['/focustrain/'],
      meta: {
        title: 'Focus Train',
        description: 'Focus Train is a study timer app to build deep work consistency.',
        canonical: 'https://www.eloantourtelier.com/focustrain/',
        favicon: '/focustrain/focustrainicon.png',
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

function upsertMetaByName(name, content) {
  if (!content) return;

  let node = document.querySelector(`meta[name="${name}"]`);
  if (!node) {
    node = document.createElement('meta');
    node.setAttribute('name', name);
    document.head.appendChild(node);
  }
  node.setAttribute('content', content);
}

function upsertMetaByProperty(property, content) {
  if (!content) return;

  let node = document.querySelector(`meta[property="${property}"]`);
  if (!node) {
    node = document.createElement('meta');
    node.setAttribute('property', property);
    document.head.appendChild(node);
  }
  node.setAttribute('content', content);
}

function upsertCanonical(href) {
  if (!href) return;

  let node = document.querySelector('link[rel="canonical"]');
  if (!node) {
    node = document.createElement('link');
    node.setAttribute('rel', 'canonical');
    document.head.appendChild(node);
  }
  node.setAttribute('href', href);
}

router.afterEach((to) => {
  const title = to.meta.title || 'Eloan Tourtelier';
  const description = to.meta.description || 'Engineering student based in Paris.';
  const canonical = to.meta.canonical || 'https://www.eloantourtelier.com/';
  document.title = title;

  const link = document.querySelector("link[rel~='icon']");
  if (link) {
    link.href = to.meta.favicon || '/homepage/favicon.jpeg';
  }

  upsertMetaByName('description', description);
  upsertMetaByName('robots', 'index, follow');
  upsertMetaByName('googlebot', 'index, follow, max-snippet:-1, max-image-preview:large, max-video-preview:-1');
  upsertMetaByProperty('og:type', 'website');
  upsertMetaByProperty('og:title', title);
  upsertMetaByProperty('og:description', description);
  upsertMetaByProperty('og:url', canonical);
  upsertCanonical(canonical);
});

export default router;
