This folder is for pm2 workflow configuration

## How to use

1. install pm2
2. start task

```bash
npm i -g pm2
pm2 start task.config.js
```

This enables background task running. You can check the status by `pm2 list` and `pm2 logs`.
