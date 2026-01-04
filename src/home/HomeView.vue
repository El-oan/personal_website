<template>
  <nav class="sidebar" aria-label="Section navigation">
    <div class="side-inner">
      <div class="side-group">
        <div class="side-label">Me</div>
        <a class="side-link" href="#about">About</a>
      </div>

      <div class="side-group">
        <div class="side-label">Work</div>
        <a class="side-link" href="#projects">Projects</a>
        <a class="side-link" href="#experiences">Experiences</a>
        <a class="side-link" href="#education">Education</a>
      </div>

      <div class="side-group">
        <div class="side-label">Contact</div>
        <a class="side-link" href="#connect">Reach Out</a>
      </div>
    </div>
  </nav>

  <div class="container">
    <h1 class="page-title">Eloan Tourtelier</h1>
    <p class="subtitle">Engineering Student</p>

    <div id="about" class="section">
      <p>
        Hi! I'm a 22-year-old engineering student at
        <strong>CentraleSupélec Paris-Saclay</strong>, graduating in 2027. I am currently based in Paris.
      </p>
      <p>
        I love design, languages and coffee shops. I speak English and French, and I'm learning Chinese (Russian too but
        if I start a russian sentence I end up speaking chinese). I recently discovered I really enjoy machine learning too.
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
          style="width: 15%; border-radius: 10px; margin-bottom: 12px; border: 1px solid var(--border-color)"
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
        <div class="card-title">Attention Implementation</div>
        <div class="card-meta">Machine Learning • Work in progress</div>
        <div class="card-desc">
          An implementation of the Transformer architecture from almost scratch, 
          following the <em>Attention Is All You Need</em> paper, with some adjustements. 
          Uses multi-head attention, positional encoding, residual connections, 
          Layer Normalization. Trained on Kaggle's GPUs.
          Work in progress.
        </div>
        <div class="tags-row">
          <div class="tags">
            <span class="tag">Python</span>
            <span class="tag">PyTorch</span>
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
        <a href="https://www.instagram.com/eloan.trtl/" target="_blank" class="link-button"> Instagram </a>
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
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted } from 'vue';

let cleanup;

onMounted(() => {
  const links = Array.from(document.querySelectorAll('.side-link'));
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
  if (cleanup) cleanup();
});
</script>


