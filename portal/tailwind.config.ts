import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'carbon': '#254b1a',
        'carbon-light': '#3d7a2a',
        'carbon-dark': '#1a3612',
      },
    },
  },
  plugins: [],
}
export default config
