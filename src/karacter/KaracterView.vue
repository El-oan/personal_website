<template>
  <div class="app-wrapper">
    <nav class="navbar">
      <div class="container nav-content">
        <div class="brand">
          <img src="/GreenDarkLogo.png" alt="Karacter Logo" class="nav-logo" />
          <span class="nav-title">Karacter</span>
        </div>
        <a href="/" class="back-link">← eloantourtelier.com</a>
      </div>
    </nav>

    <main class="container">
      <section class="hero fade-in-up">
        <img src="/GreenDarkLogo.png" alt="Karacter App Icon" class="app-icon floating" />
        <h1 class="hero-title">Master Chinese Offline</h1>
        <p class="hero-subtitle">
          The essential companion for Chinese learners. Handwriting recognition, HSK vocabulary, and AI sentence breakdown—all without internet.
        </p>
        
        <div class="cta-group">
          <a href="https://apps.apple.com/us/app/%E6%96%87-character/id6747664971?l=zh-Hans-CN" target="_blank" class="btn btn-primary pulse-hover">
            Download on App Store
          </a>
          <a href="mailto:character.help@gmail.com" class="btn btn-secondary">
            Contact Support
          </a>
        </div>
      </section>

      <section class="features-section">
        <h2 class="section-title">Why Karacter?</h2>
        <div class="features-list">
          <div class="feature-item slide-in-left">
            <div class="feature-icon-wrapper">
              <span class="feature-icon">📚</span>
            </div>
            <div class="feature-content">
              <h3>Offline Dictionary</h3>
              <p>Search words, pinyin, or English definitions instantly. No VPN or internet required—perfect for travel or commute.</p>
            </div>
          </div>
          
          <div class="feature-item slide-in-right">
            <div class="feature-content text-right-md">
              <h3>Handwriting</h3>
              <p>Draw characters directly on screen. Our recognition engine helps you find meanings and practice correct stroke order.</p>
            </div>
            <div class="feature-icon-wrapper">
              <span class="feature-icon">✍️</span>
            </div>
          </div>

          <div class="feature-item slide-in-left">
            <div class="feature-icon-wrapper">
              <span class="feature-icon">🧠</span>
            </div>
            <div class="feature-content">
              <h3>AI Analysis</h3>
              <p>Stuck on a complex sentence? The built-in AI breaks it down into understandable grammar blocks.</p>
            </div>
          </div>

          <div class="feature-item slide-in-right">
            <div class="feature-content text-right-md">
              <h3>HSK 3.0 Ready</h3>
              <p>Stay up to date. All vocabulary is labeled with the latest HSK levels so you can track your true proficiency.</p>
            </div>
            <div class="feature-icon-wrapper">
              <span class="feature-icon">📈</span>
            </div>
          </div>
        </div>
      </section>

      <section class="history-section fade-in">
        <div class="history-content">
          <div class="history-text">
            <h2>The Story Behind Karacter</h2>
            <p>
              I started building Karacter because I was frustrated with existing Chinese learning apps. 
              As a student, I needed a tool that was fast, clean, and most importantly—<strong>offline</strong>.
            </p>
            <p>
              Many dictionary apps required a constant internet connection or were cluttered with ads. 
              I wanted something simple that just worked, whether I was in a subway tunnel or a classroom with spotty Wi-Fi.
            </p>
            <p>
              What began as a personal project with this original logo (right) has grown into a full-featured iOS app 
              helping thousands of other learners master the language.
            </p>
          </div>
          <div class="history-image-wrapper">
            <img src="/OldLogo.png" alt="Original Karacter Logo" class="history-image" />
            <span class="image-caption">The original logo design</span>
          </div>
        </div>
      </section>

      <section class="roadmap-section">
        <div class="roadmap-container">
          <h2>Coming Soon</h2>
          <div class="roadmap-grid">
            <div class="roadmap-card">
              <span class="roadmap-icon">🌍</span>
              <h4>Multi-language</h4>
              <p>Support for French, Spanish, and German interfaces.</p>
            </div>
            <div class="roadmap-card">
              <span class="roadmap-icon">🔍</span>
              <h4>Advanced Filters</h4>
              <p>Search by HSK level, radical, or stroke count.</p>
            </div>
            <div class="roadmap-card">
              <span class="roadmap-icon">📇</span>
              <h4>Flashcards</h4>
              <p>Spaced repetition system to help you retain vocabulary.</p>
            </div>
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
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
// No script logic needed for now
</script>

<style scoped>
/* Scoped overrides if needed */
</style>
