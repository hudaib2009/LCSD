FROM node:20-bookworm-slim

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm ci --no-audit --no-fund

COPY frontend ./
RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start"]
