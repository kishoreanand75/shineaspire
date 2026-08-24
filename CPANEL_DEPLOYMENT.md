# cPanel Deployment

This project has two different runtime surfaces:

- `passenger_wsgi.py` serves the live dashboard through cPanel Passenger.
- `main.py` is a separate long-running paper-trading worker and must not be
  started by a web request.
- In the default configuration, paper mode attempts one simulated entry per
   5-minute slot and keeps the existing 20-minute exit/risk rules.

## Create the cPanel Python application

1. Upload the project files to a private application directory outside
   `public_html` when possible.
2. In cPanel, open **Setup Python App** and create an application with Python
   3.10 or 3.11.
3. Set **Application root** to the uploaded project directory.
4. Set **Application URL** to the required domain or subdomain.
5. Set **Startup file** to `passenger_wsgi.py`.
6. Set **Application entry point** to `application`.
7. Use **Run Pip Install** with `requirements.txt`.
8. Add the environment variables required by `config.py`, such as
   `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
9. Restart the application.

Check these URLs after restarting:

- `/health` should return `{"status":"ok"}`.
- `/` should display the Antony Quant live dashboard.

## Binance credentials

Add `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` as cPanel environment variables
only if live account verification is intentionally enabled. Do not commit them
to GitHub or upload a `.env` file containing secrets.

## Paper-trading worker

The web application does not run `main.py`. If the hosting plan provides a
reliable Python cron or process manager, run `main.py` as one persistent worker
using the hosting provider's Python executable. Do not configure multiple cron
jobs for it because it writes shared state files and uses a process lock.

If the plan only supports Passenger web requests and ordinary cron jobs, the
dashboard can be deployed, but the five-second paper-trading loop cannot be
kept alive reliably on that plan. Use a VPS or a worker-capable service for
that process.

The dashboard displays active paper entries immediately from `active_trade.json`
and completed rows after the position exits. Allow at least 20 to 25 minutes
after starting `main.py` before expecting a new completed row in `trades.csv`.