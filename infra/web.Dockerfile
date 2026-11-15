FROM node:22-alpine

WORKDIR /workspace/web

COPY web/public/package.json ./package.json
COPY web/public/package-lock.json* ./
COPY web/public/index.html ./index.html
RUN npm install

COPY web /workspace/web

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
