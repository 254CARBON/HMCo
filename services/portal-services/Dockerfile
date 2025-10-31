FROM node:lts-alpine@sha256:f36fed0b2129a8492535e2853c64fbdbd2d29dc1219ee3217023ca48aebd3787
WORKDIR /app
COPY package.json package-lock.json* /app/
RUN npm install --production --no-audit --no-fund
COPY server.js /app/
EXPOSE 8080
ENV NODE_ENV=production
CMD ["node","server.js"]
