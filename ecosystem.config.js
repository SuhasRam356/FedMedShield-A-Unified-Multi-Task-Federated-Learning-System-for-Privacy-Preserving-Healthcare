module.exports = {
  apps: [
    {
      name: "fedmedshield-backend",
      cwd: "./backend",
      script: "uvicorn",
      args: "main:app --host 0.0.0.0 --port 8000 --reload",
      interpreter: "python", 
      watch: ["."],
      ignore_watch: ["__pycache__", "*/__pycache__/*", "*.pyc"],
      env: {
        NODE_ENV: "development",
        PYTHONUNBUFFERED: "1" 
      },
      log_date_format: "YYYY-MM-DD HH:mm Z",
      error_file: "../logs/backend-error.log",
      out_file: "../logs/backend-out.log",
      merge_logs: true
    }
  ]
};
