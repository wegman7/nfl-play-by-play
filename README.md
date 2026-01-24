## start kafka server
docker compose down -v
docker compose up -d
# view kafka ui at localhost:8080

## DATA
https://github.com/nflverse/nflverse-data/releases/tag/pbp  

https://github.com/nflverse/nflfastR?tab=readme-ov-file
https://github.com/nflverse/nflverse-data

## getting actual real time data
There are free, unofficial APIs or public JSON endpoints that update every few seconds during games:
NFL’s public GameCenter JSON (not officially supported, but educational use is common)
e.g. https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard
returns current games, scores, drives, play details.
You can poll every 10–20 s and push into Kafka as “streaming” events.
SportsDataIO Free Tier
They have a developer trial that gives you 30–50 free API calls per day.
Good for testing ingestion code.
Endpoint: https://sportsdata.io/developers/api-documentation/nfl.
TheSportsDB (open REST API):
You can get live scores, teams, events (not as detailed as play-by-play).
Free with API key, no credit card.
College football APIs (like cfbd): also free and structured similarly.

💸 4. What’s actually free vs. paid
Source	Live?	Cost	Granularity	Notes
nflfastR	❌ (historical)	Free	Play-by-play	Perfect for model training
ESPN/NFL JSON endpoints	✅ (limited live)	Free	Drives & scores	Unofficial but usable
TheSportsDB	✅	Free tier	Team & score	Simple to use
SportsDataIO	✅	Free dev tier	Full play detail	Daily call limits
Sportradar	✅	$$$	Enterprise live feed	Real-time, licensed
Synthetic / simulator	✅ (your control)	Free	Anything	Perfect for streaming practice

🧠 Recommended approach for your NFL MLOps project
Train models on historical nflfastR data.
Simulate live events (your own Kafka producer from that data).
Once your pipeline (Kafka → Spark → Feature Store → MLflow → FastAPI) works locally,
swap in one of:
ESPN JSON scoreboard (for near-live),
or SportsDataIO free API for small batches,
or, later, Sportradar if you want true production-grade.
This lets you master streaming + ML orchestration with zero API fees.

## model potential features:
A) Game-state features (always available)  
quarter,  
clock_remaining_seconds,  
score_diff (home − away),  
possession_team,  
is_home_possession,  
down,  
distance,  
yardline_100 (distance to end zone)
timeouts_home,  
timeouts_away,  
home/away indicator,  
neutral_site,  
posteam_time_in_possession (rolling)  

B) Team strength & priors (pregame context you carry through the game)
team_strength_home, team_strength_away (e.g., Elo/Glicko, or last-N EPA)
vegas_spread_closing, total_closing (if available)
Rolling form: last 3 games offensive/defensive EPA/play, success rate  

C) On-field tendencies (rolling, online-updated)
run_rate_last_10_plays, pass_rate_last_10_plays
qb_completion_rate_trailing_10, pressure_rate_allowed_trailing_50
explosive_play_rate (≥ 20 yards) trailing window
red_zone_efficiency trailing window  

D) Environment
roof (indoor/outdoor), surface, temperature, wind, precip_flag
All rolling features must be online-safe: only use events up to t.



## building pregame model
9) Can the same model do pregame WP?
Short answer: you can, but a separate pregame model is better.
Option A — Reuse the in-game WP model
Construct the initial game state:
quarter=1, clock_remaining=3600, score_diff=0, yardline_100=75 (typical kickoff), timeouts_home=3, timeouts_away=3.
Include pregame priors such as team_strength_home/away, vegas_spread, home_field.
Feed that into the same model to get a starting WP.
Caveat: If you trained mostly on in-game states, the model’s calibration at t=0 can be off (there are far fewer early-game states than mid/late).
Option B — Train a dedicated Pregame WP model (recommended)
Use only pregame features: team strengths, injuries, home field, weather, Vegas spread/total.
Label = game outcome (home win).
Output = pregame WP.
Blend it with in-game WP early on, e.g.:
WP_blended = α(t) * WP_pregame + (1-α(t)) * WP_ingame, where α decays from 1 → 0 during Q1.
This produces excellent early-game calibration and smooth transitions.


## Minimal feature list (to start strong)
State: quarter, clock_seconds, score_diff, down, distance, yardline_100, timeouts_home/away, is_home_possession
Priors: team_strength_home/away (Elo), vegas_spread, home_field
Rolling: offense_epa_l50, defense_epa_allowed_l50, pass_rate_l10, qb_comp_l10
Env: roof/surface, temp, wind


## to install airflow
pip install "apache-airflow[celery]==3.1.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.2/constraints-3.13.txt"



### TODO
 - add a separate model for pregame predictions
    - use a combination of this model + my play by play model for in game predictions
 - look into best practices for model staging/promotion
 - work on making model better: look into more feature engineering, looking at different model architectures, which one makes the most sense and why. also look into xgboost and hyperparameter searching
 
 - incorporate pregame betting lines and totals in model to improve accuracy
 - start research on correlation between betting lines and results
  - eg does always betting dogs/favorites/home/away perform better?
  - how accurate are betting lines in general?

 - FIX BUG: TOTAL_HOME_SCORE AND TOTAL_AWAY_SCORE ARE SCORES AT THE ENDDDD OF THE PLAY, USE POSTEAM_SCORE AND DEFTEAM_SCORE INSTEAD (AND WE WON'T HAVE TO DO THE HOME->POST TRANSFORMATION)
