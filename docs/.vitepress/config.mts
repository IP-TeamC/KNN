import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  lang: 'de-DE',
  title: "KNN Docs",
  description: "Dokumentation zur KNN-Bibliothek",

  base: '/KNN/',

  lastUpdated: true,
  cleanUrls: true,

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Startseite', link: '/' },
      { text: 'About', link: '/about' },
    ],

    sidebar: [
      {
        text: 'Einführung',
        items: [
          { text: 'Getting Started', link: '/overview' }
        ]
      },
      {
        text: 'Dokumentation',
        items: [
          { text: 'Bibliothek', link: '/library' },
          { text: 'Konfiguration', link: '/configuration' },
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/IP-TeamC/KNN' }
    ],

    footer: {
      message: 'Veröffentlicht unter der ISC Lizenz.',
      copyright: 'Copyright © 2026 Marcel Anker, Lennart Heinrich, Piet Ostendorp'
    },

    lastUpdated: {
      text: 'Letzte Änderung'
    },

    docFooter: {
      prev: 'Vorherige Seite',
      next: 'Nächste Seite'
    },

    lightModeSwitchTitle: 'Wechsel zu hellem Modus',
    darkModeSwitchTitle: 'Wechsel zu dunklem Modus',
    sidebarMenuLabel: 'Menü',
    returnToTopLabel: 'Nach oben',
    langMenuLabel : 'Sprachauswahl',
    skipToContentLabel: 'Zum Inhalt springen',

    outline: { level: [2, 3], label: 'Auf dieser Seite' }
  },

  vite: {
    plugins: [
      {
        name: 'redirect-plugin',
        configureServer(server) {
          server.middlewares.use((req, res, next) => {
            const url = (req as any).url as string
            if (url?.startsWith('/docs/')) {
              const newUrl = url.replace('/docs/', '/')
              res.writeHead(301, { Location: newUrl })
              res.end()
              return
            }
            next()
          })
        }
      }
    ]
  },
  rewrites: {
    'docs/:page': ':page'
  }
})
