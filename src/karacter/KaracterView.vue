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
      </section>

      <section class="history-section">
        <div class="history-content">
          <div class="history-text">
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

      <section class="history-section">
        <div class="history-content">
          <div class="history-text">
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
          <div class="history-image-wrapper">
            <img src="/OldLogo.png" alt="Original Karacter Logo" class="history-image" />
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
        </div>
      </div>
    </footer>
  </div>
</template>
