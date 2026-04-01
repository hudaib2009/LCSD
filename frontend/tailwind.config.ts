import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#101418",
        mist: "#f4f7f9",
        sea: "#0f6b78",
        sun: "#f3b43f",
        alert: "#cc3d3d",
        slate: "#344056"
      },
      boxShadow: {
        soft: "0 20px 50px -30px rgba(12, 38, 55, 0.45)"
      },
      backgroundImage: {
        "radial-glow": "radial-gradient(circle at top, rgba(243, 180, 63, 0.35), transparent 55%)"
      }
    }
  },
  plugins: []
};

export default config;
