This page contains instructions about how to setup Claude over Slack for the evaluation. In essence, we will add Claude app into your Slack workspace, and set up another Slack app which the scripts will use to interact with the Claude app. 

Please follow these steps:

1. Log in a *paid* slack workspace on browser.
1. Open [https://www.anthropic.com/claude-in-slack](https://www.anthropic.com/claude-in-slack) and click "Add to Slack" so as to add Claude app into the Slack workspace.
1. Open [https://api.slack.com/apps](https://api.slack.com/apps) and click "Create New App" to create a Slack app. 
1. Open "OAuth & Permissions" tab, and in "User Token Scopes" add the following permissions:
    * admin
    * channels:history
    * channels:read
    * channels:write
    * chat:write
    * groups:history
    * groups:read
    * groups:write
    * im:history
    * im:read
    * im:write
    * mpim:history
    * mpim:read
    * mpim:write
    * users:read
1. Click "Reinstall to Workspace" so that the permission changes are applied
1. Copy your "OAuth Tokens for Your Workspace" starting with "xoxp-" and execute `export SLACK_USER_TOKEN=<token>` in the command line.