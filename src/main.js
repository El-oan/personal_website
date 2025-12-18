import { createApp } from 'vue';
import App from './App.vue';
import router from './router';

import './home/home.css';
import './karacter/karacter.css';

const app = createApp(App);

app.use(router);

app.mount('#app');

