<template>
  <div class="home-wrapper" :class="{ 'nav-open': isNavOpen }">
    <canvas 
      ref="bgCanvas" 
      class="bg-canvas"
      style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0;"
    ></canvas>
    <nav
      class="sidebar"
      :class="{ 'is-open': isNavOpen }"
      aria-label="Section navigation"
      @mouseenter="openNav"
      @mouseleave="closeNav"
      @focusin="openNav"
      @focusout="handleNavFocusOut"
    >
      <div class="side-inner">
        <a class="side-item" href="#about">
          <span class="side-bar" aria-hidden="true"></span>
          <span class="side-text">About</span>
        </a>
        <a class="side-item" href="#projects">
          <span class="side-bar" aria-hidden="true"></span>
          <span class="side-text">Projects</span>
        </a>
        <a class="side-item" href="#experiences">
          <span class="side-bar" aria-hidden="true"></span>
          <span class="side-text">Experiences</span>
        </a>
        <a class="side-item" href="#education">
          <span class="side-bar" aria-hidden="true"></span>
          <span class="side-text">Education</span>
        </a>
        <a class="side-item" href="#connect">
          <span class="side-bar" aria-hidden="true"></span>
          <span class="side-text">Reach Out</span>
        </a>
      </div>
    </nav>
    <div class="blur-scrim" aria-hidden="true"></div>

    <div class="home-container">
      <h1 class="page-title">Eloan Tourtelier</h1>
      <p class="subtitle">Engineering Student</p>

      <div id="about" class="section">
        <p>
          Hi! I'm a 22-year-old engineering student at
          <strong>CentraleSupélec Paris-Saclay</strong>, graduating in 2027. 
          I am currently based in Paris.
        </p>
        <p>
          I love design, languages and coffee shops. 
          I speak English and French, and I'm learning Chinese (Russian too but
          if I start a russian sentence I end up speaking chinese). 
          I recently discovered I really enjoy machine learning too.
        </p>
        <p>
          I spent most of my life in the west of France, and 6 months in China. 
          I'd love to try the USA, and travel around the world (yep overdone ik).
          Do not hesitate one second to reach out!
        </p>
      </div>

      <div class="divider"></div>

      <div id="projects" class="section">
        <h2>Projects</h2>

        <div class="card">
          <img
            src="/KaracterLogo.png"
            alt="Karacter logo"
            class="card-logo"
          />
          <div class="card-title">Karacter — Chinese Learning App</div>
          <div class="card-meta">iOS • Live on the App Store</div>
          <div class="card-desc">
            An offline Chinese learning helper with character drawing recognition, HSK 3.0 labeling, AI-powered sentence
            analysis. You can search for any Chinese word, expression, pinyin, English term — or hand-draw any character
            you want to learn more about. Pronunciations are available, as well as the correct strokes order. No internet
            connection nor VPN required.
          </div>
          <div class="tags-row">
            <div class="tags">
              <span class="tag">Swift</span>
              <span class="tag">SQLite</span>
              <span class="tag">XCode</span>
              <span class="tag">Coffee</span>
            </div>
            <a class="card-button" href="/karacter/">文 website</a>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Attention Network</div>
          <div class="card-meta">Machine Learning • Work in progress</div>
          <div class="card-desc">
            An implementation of the Transformer architecture, following the 
            <em>Attention Is All You Need</em> paper, with some improvements. 
            Uses multi-head attention, positional encoding, Deepseek's 
            mHC connections, RMS normalization. Trained on Kaggle's GPUs.
            Work in progress.
          </div>
          <div class="tags-row">
            <div class="tags">
              <span class="tag">Python</span>
              <span class="tag">PyTorch</span>
              <span class="tag">Jupyter</span>
            </div>
            <a class="card-button" href="/notebook">notebook</a>
          </div>
        </div>

        <div class="card">
        <div class="card-title">Mistral Hackathon</div>
        <div class="card-meta">Weekend Competition • September 2025 • 1500+ LOC</div>
        <div class="card-desc">
          Competed in a Mistral AI hackathon. Discovered and developed Model Context
          Protocol (MCP) connectors enabling LLMs to call external tools and pipelines. Built a demo using Spotify,
          Genius, and Wikipedia APIs to add music tooling to "Le Chat" AI model.
        </div>
        <div class="tags-row">
          <div class="tags">
            <span class="tag">MCP</span>
            <span class="tag">LLM Tooling</span>
          </div>
        </div>
        </div>
        
      </div>


      <div class="divider"></div>

      <div id="experiences" class="section">
        <h2>Experiences</h2>

        <div class="card">
          <div class="card-title">Data Internship</div>
          <div class="card-meta">Forvis Mazars • Paris • sep 2025 - feb 2026</div>
          <div class="card-desc">Worked as a data Analyst/Engineer/Devops.</div>
        </div>

        <div class="card">
          <div class="card-title">Communication lead, Mountain Club</div>
          <div class="card-meta">CentraleSupélec • 2024 - 2025</div>
          <div class="card-desc">Managed the club's communication and image.</div>
        </div>

        <div class="card">
          <div class="card-title">Sponsorship lead, Bouldering Club</div>
          <div class="card-meta">CentraleSupélec • 2024 - 2025</div>
          <div class="card-desc">Approached companies to establish partnerships.</div>
        </div>
      </div>

      <div class="divider"></div>

      <div id="education" class="section">
        <h2>Education</h2>

        <div class="card">
          <div class="card-title">CentraleSupélec</div>
          <div class="card-meta">2023 - 2027 • Paris-Saclay</div>
          <div class="card-desc">
            MEng at CentraleSupélec, 1st European Engineering School (Shanghai ranking). Ranked 100th/6000+ in national
            entrance exam. Advanced Mathematics, Machine Learning, Time Series Econometrics, Quantum physics, etc. I hate this school.
          </div>
        </div>

        <div class="card">
          <div class="card-title">Tongji University</div>
          <div class="card-meta">2025 • Shanghai</div>
          <div class="card-desc">Exchange program at 同济大学 for 6 months. Civil Engineering.</div>
        </div>

        <div class="card">
          <div class="card-title">Lycée Louis-le-Grand</div>
          <div class="card-meta">2022 - 2023 • Paris</div>
          <div class="card-desc">
            Classes Préparatoires. Intensive study of mathematics, physics, and chemistry.
          </div>
        </div>
      </div>

      <div class="divider"></div>

      <div id="connect" class="section">
        <h2>Connect</h2>
        <p>Feel free to reach out :)</p>
        <div class="links">
          <a href="https://www.linkedin.com/in/eloantourtelier/" target="_blank" class="link-button"> LinkedIn </a>
          <a href="https://github.com/El-oan" target="_blank" class="link-button"> GitHub </a>
          <a href="https://x.com/eeloannn" target="_blank" class="link-button"> Twitter </a>
          <a href="/resume.pdf" class="link-button" download> Resume (PDF) </a>
          <a href="https://www.kaggle.com/eloantourtelier" target="_blank" class="link-button"> Kaggle </a>
          <a href="https://apps.apple.com/us/app/%E6%96%87-character/id6747664971" target="_blank" class="link-button">
            Download Karacter
          </a>
          <a href="https://sites.google.com/view/characteres/accueil" target="_blank" class="link-button">
            Karacter Website
          </a>
        </div>
      </div>

      <div class="footer">
        <p>
          Favicon designed by <a href="https://readymag.website/u1139024403/5218555/?utm_source=ig&utm_medium=social&utm_content=link_in_bio&fbclid=PAZXh0bgNhZW0CMTEAc3J0YwZhcHBfaWQMMjU2MjgxMDQwNTU4AAGn_c3mcIRik0vZ08Jpc-bp6XNRk47TOwt_ZzAqnGqCemrG5qnD6XUge_j3lsc_aem__m7zC50WEINizWnxMA-ShA" target="_blank" rel="noopener noreferrer">my sister</a>
        </p>
        <img src="/favicon.jpeg" alt="Favicon" class="footer-icon" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onBeforeUnmount, onMounted } from 'vue';

let cleanup;

const bgCanvas = ref(null);
const isNavOpen = ref(false);

function openNav() {
  isNavOpen.value = true;
}

function closeNav() {
  isNavOpen.value = false;
}

function handleNavFocusOut(event) {
  const nextTarget = event.relatedTarget;
  if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) {
    closeNav();
  }
}

onMounted(() => {
  document.body.style.backgroundColor = '#121212';

  const canvas = bgCanvas.value;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width, height;
  let animationFrame;

  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
  };
  
  window.addEventListener('resize', resize);
  resize();

  const particles = [];
  const spacing = 30; // Distance between dots
  
  const render = (time) => {
    ctx.clearRect(0, 0, width, height);
    
    const t = time * 0.001;
    
    const rows = Math.ceil(height / spacing);
    const cols = Math.ceil(width / spacing);

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        // Primary Grid Point
        const px = x * spacing + (y % 2 === 0 ? 0 : spacing / 2);
        const py = y * spacing;

        const val = Math.sin(x * 0.1 + t) * 0.5 + 
                    Math.sin(y * 0.15 + t * 0.8) * 0.5 + 
                    Math.sin((x + y) * 0.05 + t * 0.3) * 0.3 +
                    Math.sin((x - y * 1.5) * 0.03 + t * 0.5) * 0.2;
                    
        const norm = (val + 1.5) / 3; 
        
        // Binary threshold: sudden appearance
        if (norm > 0.60) {
          ctx.fillStyle = `rgba(84, 104, 255, 0.4)`; // Blue
          ctx.fillRect(px, py, 2.9, 2.9);
        }

        // Secondary Grid Point (Independent)
        const px2 = x * spacing + (y % 2 === 0 ? spacing / 2 : 0);
        const py2 = y * spacing; 

        // Different frequencies and directions for independence
        const val2 = Math.cos(x * 0.12 - t * 0.8) * 0.5 + 
                     Math.sin(y * 0.18 + t * 1.1) * 0.5 + 
                     Math.cos((x - y) * 0.04 + t * 0.4) * 0.3 +
                     Math.sin((x + y * 1.2) * 0.04 - t * 0.6) * 0.2;

        const norm2 = (val2 + 1.5) / 3;

        // Binary threshold for second grid
        if (norm2 > 0.60) {
          ctx.fillStyle = `rgba(65, 80, 210, 0.5)`; // Lighter Blue
          ctx.fillRect(px2, py2, 2, 2);
        }
      }
    }
    
    animationFrame = requestAnimationFrame(render);
  };

  animationFrame = requestAnimationFrame(render);

  // Store for cleanup
  canvas._cleanup = () => {
    window.removeEventListener('resize', resize);
    cancelAnimationFrame(animationFrame);
  };

  const links = Array.from(document.querySelectorAll('.side-item'));
  const ids = links.map((a) => a.getAttribute('href')).filter((h) => h && h.startsWith('#'));

  const sections = ids.map((id) => document.querySelector(id)).filter(Boolean);

  function activate(id) {
    links.forEach((l) => l.classList.toggle('active', l.getAttribute('href') === id));
  }

  const pickActive = () => {
    const y = 120; 
    let best = sections[0];
    for (const s of sections) {
      const top = s.getBoundingClientRect().top;
      if (top <= y) best = s;
      else break;
    }
    if (best?.id) activate(`#${best.id}`);
  };

  let raf = 0;
  const onScroll = () => {
    if (raf) return;
    raf = window.requestAnimationFrame(() => {
      raf = 0;
      pickActive();
    });
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  pickActive();

  cleanup = () => {
    window.removeEventListener('scroll', onScroll);
    window.removeEventListener('resize', onScroll);
    if (raf) window.cancelAnimationFrame(raf);
  };
});

onBeforeUnmount(() => {
  document.body.style.backgroundColor = '';
  
  if (bgCanvas.value && bgCanvas.value._cleanup) {
    bgCanvas.value._cleanup();
  }
  
  if (cleanup) cleanup();
});
</script>

