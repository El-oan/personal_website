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
        <h1 class="hero-title">Master Chinese Offline</h1>
        <p class="hero-subtitle">
          The essential companion for Chinese learners. Handwriting recognition, HSK vocabulary, and AI sentence breakdown—all without internet.
        </p>
        
        <div class="cta-group">
          <a href="https://apps.apple.com/us/app/%E6%96%87-character/id6747664971?l=zh-Hans-CN" target="_blank" class="btn btn-primary">
            Download on App Store
          </a>
          <a href="mailto:character.help@gmail.com" class="btn btn-secondary">
            Contact Support
          </a>
        </div>
      </section>

      <section class="history-section">
        <div class="history-content">
          <div class="history-text">
            <h2>Why Karacter?</h2>
            <p>
              Karacter is built to be <strong>fast</strong>, <strong>clean</strong>, and <strong>fully offline</strong>.
              Search Chinese words, pinyin, or English—anytime, anywhere, without a VPN or internet connection.
            </p>
            <p>
              If you don’t know a character, you can <strong>hand-draw</strong> it to find it instantly. You can also
              listen to pronunciations, save vocabulary for later, and follow proper stroke order.
            </p>
            <p>
              For harder sentences, Karacter includes <strong>AI features</strong> that break down complex structures
              into understandable blocks. Vocabulary is labeled using the official <strong>HSK 3.0</strong> list.
            </p>
            <div class="pill-row">
              <span class="pill">Offline</span>
              <span class="pill">Handwriting</span>
              <span class="pill">AI</span>
              <span class="pill">HSK 3.0</span>
              <span class="pill">Stroke Order</span>
            </div>
          </div>
          <div class="history-image-wrapper">
            <div class="why-mark">文</div>
            <span class="image-caption">wén — “character / writing”</span>
          </div>
        </div>
      </section>

      <section class="history-section">
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
