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
        <h1 class="hero-title">Pocket Chinese Offline</h1>
        <p class="hero-subtitle">
          An essential companion for Chinese learners. Handwriting recognition, HSK levels, 
          dictionary definitions, stroke order — all without an internet connection        
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
          <div class="stat-card">

            
          </div>
        </div>
      </section>

      <section class="content-section">
        <div class="feature-block">
          <div class="feature-text">
            <h2>What is Karacter?</h2>
            <p>
              Karacter is built to be <strong>fast</strong>, <strong>clean</strong>, and <strong>fully offline</strong>. 
              Only the AI-sentences breakdown requires an internet connection.
              Search Chinese words, pinyin, or English — anytime, anywhere, without a VPN or internet connection.
              Karacter is <strong>free</strong> and will remain free.
            </p>
            <p>
              If you don’t know a character, you can <strong>hand-draw</strong> it to find it instantly. You can also
              listen to pronunciations, save vocabulary for later, and see the proper <strong> stroke order</strong>.
            </p>
            <p>
              For longer sentences, Karacter includes <strong>AI features</strong> that break down complex structures
              into understandable blocks. Vocabulary is labeled using the latest official <strong>HSK 3.0</strong> convention.
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
