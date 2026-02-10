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
        <img src="/KaracterLogo.png" alt="Karacter App Icon" class="app-icon" />
        <h1 class="hero-title">Karacter - define chinese</h1>
        <p class="hero-subtitle">
          A simple and powerful toolkit for learning and navigating the Chinese language. 
          The reliable companion that works anywhere in the world.
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
              <div class="stat-label">High Quality Definitions</div>
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-content">
              <div class="stat-value counter" data-target="9500">0</div>
              <div class="stat-label">Vectorized Stroke Animations</div>
            </div>
          </div>
        </div>
      </section>

      <section class="content-section bento-section">
        
        <div class="bento-grid">
          <!-- Large Card: Offline Dictionary -->
          <div class="bento-card card-large fade-in-up">
            <div class="bento-card-content">
              <h3>100% Offline Dictionary</h3>
              <p>Requires no internet or VPN, making it the perfect tool for traveling through China or studying without distractions.</p>
              <div class="bento-card-visual">
                <img src="/entrycard.png" alt="Karacter Entry Card" class="bento-image dictionary-image" />
              </div>
            </div>
          </div>

          <!-- Small Card: Handwriting -->
          <div class="bento-card card-small fade-in-up" style="animation-delay: 0.1s;">
            <div class="bento-card-content">
              <h3>Handwriting & Search</h3>
              <p>Instantly find words by drawing them.</p>
            </div>
          </div>

          <!-- Small Card: HSK 3.0 -->
          <div class="bento-card card-small fade-in-up" style="animation-delay: 0.2s;">
            <div class="bento-card-content">
              <h3>HSK 3.0</h3>
              <p>Track your progress with latest standards.</p>
              <div class="bento-card-visual">
                <img src="/ancientseal.png" alt="Karacter Ancient Seal" class="bento-image hsk-image" />
              </div>
            </div>
          </div>

          <!-- Large Card: AI Sentence Breakdown -->
          <div class="bento-card card-large fade-in-up" style="animation-delay: 0.3s;">
            <div class="bento-card-content">
              <h3>AI Sentence Breakdown</h3>
              <p>Deconstructs complex phrases and explains grammar structures, acting as a bridge between a dictionary and a private tutor.</p>
              <div class="bento-card-visual">
                <img src="/entrylist.png" alt="Karacter Entry List" class="bento-image ai-image" />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="content-section fade-in">
        <div class="feature-block">
          <div class="feature-text">
            <h2>What is Karacter?</h2>
            <p>
              Karacter is a simple and powerful toolkit for learning and navigating the Chinese language. 
              It is designed for students, travelers, and enthusiasts who need a reliable companion 
              that works anywhere in the world.
            </p>

            <p>
              - A 100% offline official dictionary that requires no internet or VPN, making it the perfect tool for traveling through China or studying without distractions. <br>
              - Advanced handwriting and search where you can instantly find words by drawing them, or searching via Pinyin, English, and Chinese characters. <br>
              - AI-powered sentence breakdown that deconstructs complex phrases and explains grammar structures, acting as a bridge between a dictionary and a private tutor. <br>
            </p>

            <p>
              Karacter is a great alternative to Pleco, Hanping, and HelloChinese for those seeking 
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
              process of being a rat employee.
            </p>
          </div>
          <div class="feature-image-wrapper">
            <img src="/OldLogo.png" alt="Original Karacter Logo" class="feature-image" />
            <span class="image-caption">The first logo design. But it looked too similar to Alipay. I made it with Illustrator.</span>
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
    <div class="qr-float-card">
      <img src="/QRCode.png" alt="Download on App Store" class="qr-code-img" />
      <div class="qr-content">
        <div class="qr-footer-row">
            <img src="/appstore.png" alt="Available on App Store" class="qr-store-badge-img" />
        </div>
      </div>
      <a href="https://apps.apple.com/us/app/%E6%96%87-character/id6747664971?l=zh-Hans-CN" target="_blank" class="qr-link-overlay"></a>
    </div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue';

onMounted(() => {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        
        // Trigger counters if it's a counter element
        if (entry.target.classList.contains('counter')) {
          animateCounter(entry.target);
          observer.unobserve(entry.target); // Only animate once
        }
      }
    });
  }, { threshold: 0.1 });

  // Use a timeout to ensure DOM is ready and animations don't conflict with load
  setTimeout(() => {
      document.querySelectorAll('.fade-in, .fade-in-up').forEach(el => observer.observe(el));
      // Observe counters specifically
      document.querySelectorAll('.counter').forEach(el => observer.observe(el));
  }, 100);
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
      else el.innerText = target;
      
      clearInterval(timer);
    } else {
      // Intermediate formatting
      if (target === 178) el.innerText = current + 'k+';
      else if (target === 9500) el.innerText = current + '+';
      else el.innerText = current;
    }
  }, 1000 / fps);
}
</script>
