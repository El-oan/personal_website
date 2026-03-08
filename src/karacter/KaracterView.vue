<template>
  <div class="app-wrapper">
    <nav class="navbar">
      <div class="container nav-content">
        <div class="brand">
          <span class="nav-title">文 - Karacter</span>
        </div>
        <RouterLink to="/" class="back-link">← home</RouterLink>
      </div>
    </nav>

    <main class="container">
      <section class="hero">
        <img src="/karacter/KaracterLogo.png" alt="Karacter App Icon" class="app-icon" />
        <h1 class="hero-title">Karacter - define chinese</h1>
        <p class="hero-subtitle">
          A beautiful and powerful toolkit for learning and navigating the Chinese language. 
          A reliable companion that works anywhere in the world.
        </p>
        
        <div class="cta-group">
          <a href="https://apps.apple.com/us/app/%E6%96%87-character/id6747664971?l=zh-Hans-CN" target="_blank" class="btn btn-primary">
            Download on the App Store
          </a>
          <a href="mailto:character.help@gmail.com" class="btn btn-secondary">
            Contact us
          </a>
        </div>

        <div class="stats-grid fade-in-up" style="animation-delay: 0.15s;">
          <div class="stat-card">
            <div class="stat-content">
              <div class="stat-value counter" data-target="178">0</div>
              <div class="stat-label">high quality definitions</div>
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-content">
              <div class="stat-value counter" data-target="9500">0</div>
              <div class="stat-label">vectorized stroke animations</div>
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-content">
              <div class="stat-value counter" data-target="7200">0</div>
              <div class="stat-label">example sentences</div>
            </div>
          </div>
        </div>
      </section>

      <section class="content-section bento-section">
        
        <div class="bento-grid">
          <!-- Wide Card: Offline Dictionary -->
          <div class="bento-card card-wide card-offline card-text-only fade-in-up">
            <div class="bento-card-content">
              <h3>100% Offline Dictionary</h3>
              <p>Look up definitions, pinyin, and usage notes instantly with zero connection. Every core lookup feature works on planes, trains, and inside regions where internet access is limited.</p>
            </div>
          </div>

          <!-- Compact Card: Handwriting -->
          <div class="bento-card card-compact card-handwriting fade-in-up" style="animation-delay: 0.1s;">
            <div class="bento-card-content">
              <h3>Handwriting & Search</h3>
              <p>Draw unknown characters stroke by stroke, or search by pinyin, English, and Chinese. Built for moments when you can see a word but cannot type it yet.</p>
            </div>
          </div>

          <!-- Compact Card: HSK 3.0 -->
          <div class="bento-card card-compact card-hsk fade-in-up" style="animation-delay: 0.15s;">
            <div class="bento-card-content">
              <h3>HSK 3.0</h3>
              <p>See word levels mapped to the latest HSK standard.</p>
              <div class="bento-card-visual">
                <img src="/karacter/ancientseal.png" alt="Karacter Ancient Seal" class="bento-image hsk-image" />
              </div>
            </div>
          </div>

          <!-- Wide Card: AI Sentence Breakdown -->
          <div class="bento-card card-wide card-ai fade-in-up" style="animation-delay: 0.2s;">
            <div class="bento-card-content">
              <h3>AI Sentence Breakdown</h3>
              <p>Get line-by-line explanations for difficult sentences, including grammar roles, tone, and word choice nuances, so you understand why a phrase works, not just what it translates to.</p>
              <div class="bento-card-visual">
                <img src="/karacter/entrylist.png" alt="Karacter Entry List" class="bento-image ai-image" />
              </div>
            </div>
          </div>

          <!-- Tall Card: Lock Screen Words -->
          <div class="bento-card card-tall lockscreen-bento card-stacked fade-in-up" style="animation-delay: 0.35s;">
            <div class="bento-card-content">
              <h3>Lock Screen Words</h3>
              <p>Pin your own vocabulary to the iOS lock screen widgets and turn daily unlocks into spaced repetition, with lightweight review built into your normal phone routine.</p>
            </div>
            <div class="bento-card-image-bottom">
              <img src="/karacter/lockscreen.jpg" alt="Karacter Lockscreen Words" class="bento-image lockscreen-image" />
            </div>
          </div>
        </div>
      </section>

      <section class="content-section fade-in">
        <div class="feature-block">
          <div class="feature-text">
            <h2>What is Karacter?</h2>
            <p>
              Karacter is a minimalistic and powerful toolkit for learning and navigating the Chinese language. 
              It is designed for students, travelers, and enthusiasts who need a reliable companion 
              that works offline, anywhere in the world.
            </p>

            <p>
              1. A high quality dictionary, requiring no internet nor VPN, making it the perfect tool for traveling through China or studying without distractions. <br>
              2. Advanced handwriting and search where you can instantly find words by drawing them, searching via pinyin, English, or Chinese. <br>
              3. Homescreen and lockscreen widgets, displaying your words to see them throughout the day. <br>
              4. AI-powered explanations, deconstructing complex phrases and explaining grammar structures, usages and historical facts. <br>
            </p>

            <p>
              Karacter is a great alternative to Pleco for those seeking 
              a fast, modern, and offline-first experience!
            </p>
          </div>
        </div>
      </section>

      <section class="content-section">
        <div class="feature-block">
          <div class="feature-text">
            <h2>The Story Behind Karacter</h2>
            <p>
              When I arrived in Shanghai, I started learning Chinese. I then discovered Pleco and a 
              plethora of other Chinese learning apps, but none felt clean or intuitive to me.
              I also during this time has some free time, because Civil Engineering was lowkey boring. 
              I then learnt Swift and started iterating on Karacter.
            </p>
            <p>
              This project was mainly meant to help me learn Chinese and to feel that I could actually
              build something and exercise free will, rather than experience the soul-crushing 
              process of being an obedient cog in the wheel.
            </p>
          </div>
          <div class="feature-image-wrapper">
            <img src="/karacter/OldLogo.png" alt="Original Karacter Logo" class="feature-image" />
            <span class="image-caption">The first logo design. But it looked too similar to Alipay. I made it with Illustrator.</span>
          </div>
        </div>
      </section>

      <section ref="downloadSection" class="content-section download-section">
        <div class="download-section-card fade-in-up">
          <div class="download-copy">
            <h2>Download Karacter</h2>
            <p>
              Scan the QR code to install Karacter and keep Chinese definitions in your pocket,
              even when you are offline.
            </p>
            <a
              href="https://apps.apple.com/us/app/%E6%96%87-character/id6747664971?l=zh-Hans-CN"
              target="_blank"
              class="btn btn-primary"
            >
              Open App Store
            </a>
          </div>
          <div class="download-qr-panel">
            <img src="/karacter/QRCode.png" alt="Karacter App Store QR code" class="download-qr-image" />
            <img src="/karacter/appstore.png" alt="Available on App Store" class="download-store-badge" />
          </div>
        </div>
      </section>

    </main>

    <footer class="footer">
      <div class="container footer-content">
        <p>&copy; {{ new Date().getFullYear() }} Karacter. No personal data collected.</p>
        <div class="social-links">
          <a href="https://www.instagram.com/karacterapp/" target="_blank">Instagram</a>
          <a href="mailto:character.help@gmail.com">Email</a>
          <RouterLink to="/karacter/terms">Terms & Conditions</RouterLink>
        </div>
      </div>
    </footer>
    <div class="qr-float-card" :class="{ 'qr-float-card-hidden': isDownloadSectionVisible }">
      <img src="/karacter/QRCode.png" alt="Download on App Store" class="qr-code-img" />
      <div class="qr-content">
        <div class="qr-footer-row">
            <img src="/karacter/appstore.png" alt="Available on App Store" class="qr-store-badge-img" />
        </div>
      </div>
      <a href="https://apps.apple.com/us/app/%E6%96%87-character/id6747664971?l=zh-Hans-CN" target="_blank" class="qr-link-overlay"></a>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue';

const isDownloadSectionVisible = ref(false);
const downloadSection = ref(null);

let revealObserver;
let downloadObserver;

onMounted(() => {
  revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        
        // Trigger counters if it's a counter element
        if (entry.target.classList.contains('counter')) {
          animateCounter(entry.target);
          revealObserver.unobserve(entry.target); // Only animate once
        }
      }
    });
  }, { threshold: 0.1 });

  if (downloadSection.value) {
    downloadObserver = new IntersectionObserver((entries) => {
      isDownloadSectionVisible.value = entries.some(entry => entry.isIntersecting);
    }, { threshold: 0.35 });

    downloadObserver.observe(downloadSection.value);
  }

  // Use a timeout to ensure DOM is ready and animations don't conflict with load
  setTimeout(() => {
      document.querySelectorAll('.fade-in, .fade-in-up').forEach(el => revealObserver.observe(el));
      // Observe counters specifically
      document.querySelectorAll('.counter').forEach(el => revealObserver.observe(el));
  }, 100);
});

onUnmounted(() => {
  revealObserver?.disconnect();
  downloadObserver?.disconnect();
});

function animateCounter(el) {
  const targetStr = el.dataset.target;
  const target = parseInt(targetStr);
  
  if (isNaN(target)) return;

  const duration = 2000; // 2 seconds
  const fps = 60;
  const totalFrames = (duration / 1000) * fps;
  let frame = 0;
  
  const timer = setInterval(() => {
    frame++;
    const progress = frame / totalFrames;
    const easeOutQuad = progress * (2 - progress); // Ease out effect
    
    const current = Math.floor(easeOutQuad * target);
    
    if (frame >= totalFrames) {
      // Final formatting
      if (target === 178) el.innerText = '178k+';
      else if (target === 9500) el.innerText = '9500+';
      else if (target === 7200) el.innerText = '7200+';
      else el.innerText = target;
      
      clearInterval(timer);
    } else {
      // Intermediate formatting
      if (target === 178) el.innerText = current + 'k+';
      else if (target === 9500) el.innerText = current + '+';
      else if (target === 7200) el.innerText = current + '+';
      else el.innerText = current;
    }
  }, 1000 / fps);
}
</script>
