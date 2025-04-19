# remote-dev-ai

## Description
Remote‑Dev‑AI delivers a fully automated, AI‑driven coding assistant that integrates seamlessly into your GitHub codebase via the MCP protocol and a custom GitHub App. By labeling issues with `remote-dev-ai`, teams invoke our backend—hosted on AWS EC2 and AWS Lambda—that:

1. Fetches issue context  
2. Generates code with Anthropic Claude  
3. Opens a pull request with suggested changes  

This model targets the 75% of developer time spent on non‑coding, repetitive tasks, slashes weekly issue‑resolution costs by 70% (from \$1,200 in dev hours to \$350 in AI calls), and provides a globally scalable solution for enterprises and startups alike.

---

## Project Technology Stack

### Integration & Hosting
- **GitHub App & MCP Bot** built in Python, registered as a GitHub App. 
- Hosted on **AWS EC2** instances running our MCP server and client for persistent connections and low latency  

### Event Handling
- **AWS Lambda** functions triggered by GitHub webhooks  
- Lambda invokes the MCP server to fetch issue data and pipeline it into the Backend AI engine  

### AI Code Generation
- Anthropic accessed via secured API keys
### Front-end
- Vercel Platform


## Project Impact and Importance

- **Time Savings**  
  Developers currently waste ~75% of their workday on non‑coding tasks (documentation, debugging, manual workflows).  
- **Cost Efficiency**  
  At 100 issue calls/day, AI costs ≈\$350/week vs. \$1,200/week for a mid‑level developer at \$48.65/hr—a 70% reduction in direct labor cost.  
- **Global Scalability**  
  As a GitHub‑native solution, Remote‑Dev‑AI deploys instantly across any public or private repo, serving organizations of all sizes worldwide.  
- **Democratizing Automation**  
  Empowers small teams and emerging markets with enterprise‑grade automation without large DevOps budgets or specialist hires.  

---

## Development Challenges Faced

### 1. Immature MCP Protocol
The GitHub MCP server was open‑sourced only weeks before the hackathon, with minimal documentation. Integrating it into a production‑grade app required deep protocol analysis and custom extensions.

### 2. Context Management
Ensuring AI‑generated code aligns with each repo’s conventions demanded a dynamic context loader that fetches README, lint rules, and recent commits before each generation.(IN PROGRESS)


## TEST OUR APP:
- https://v0-remote-dev-ai.vercel.app/

## DEMO VIDEO:
https://www.youtube.com/watch?v=NuIrfwuBzAU

## TEAM: remote-dev-ai
- Yashwanth Alapati
- Rajath Reghunath
- Manan Prakashkumar Patel
- Ritvik Vasantha Kumar


## REFERENCES:
- Our front end code by vercel: https://github.com/rrgodhorus/hofhacks-remote-dev-ai-home

