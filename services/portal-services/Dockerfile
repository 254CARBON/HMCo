FROM node:lts-alpine
WORKDIR /app
COPY package.json package-lock.json* /app/
RUN npm install --production --no-audit --no-fund
COPY server.js /app/
EXPOSE 8080
ENV NODE_ENV=production
CMD ["node","server.js"]
