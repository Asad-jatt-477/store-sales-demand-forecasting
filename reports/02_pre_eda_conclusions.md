# Pre-EDA khulasa: business insights aur aage ke faisle

## Business insights (planning team ke liye)

| # | Insight | Business action |
|---|---|---|
| 1 | Weekend (Sat/Sun) sab se bare din, Thursday sab se kamzor | Thursday/Friday ko shelves bharna, weekend staff barhana |
| 2 | December saal ka peak, khas taur par 20-24 December | Christmas se 2 hafte pehle inventory barhana |
| 3 | Tankhwah ke foran baad (1-5 aur 16 tareekh, mahine ka aakhri din) sales zyada | Payday ke aas paas stock aur promotions |
| 4 | Volume chand families mein: GROCERY I, BEVERAGES, PRODUCE | Accuracy ki nigrani sab se pehle inhi families par |
| 5 | Stores ke size mein bohat bara farq; Quito aur Guayaquil haavi | Har store ka forecast uske apne level par, ek size sab par nahi |
| 6 | Promotion wale dinon par sales zyada (correlation, causation nahi) | Promotion ka asli asar naapne ke liye controlled tests |
| 7 | Holidays ke asar ulte bhi hain (+50% se -100%) | Har holiday ki planning us ke pichle asar se |
| 8 | 2013-May 2015 ka kuch data aadha record hua (BEVERAGES waghaira) | Purane reports par bharosa na karein; analysis aur training June 2015 se |
| 9 | Oil price ka sales par koi asli asar nahi dikha | Oil ko planning ka driver na samjha jaye |
| 10 | Carnaval se pehle wale Sunday sales 25-34% kam | Us Sunday ke liye kam stock aur staff |
| 11 | Naye stores pehle hafton mein zyada bechte hain, phir thehar jate hain | Opening ke waqt extra stock, phir normal |
| 12 | Forecast window mein promotion history se zyada (~44%) | Promotion ke asar ki nigrani |

## Phase 05 (feature engineering) ke faisle, sab Pre-EDA se nikle

| Feature group | Kyun (kis section se) |
|---|---|
| Sales history: lag 16/21/28/35/42, lag 364, rolling averages (16 din pehle khatam) | Section 12 (sirf lag >= 16 allowed) |
| Same-weekday averages | Section 3, 12 (weekly pattern) |
| Calendar: day of week, day of month, payday distance, month, days to Christmas | Section 3, 4, 5, 9 |
| Store/family identity aur attributes | Section 6, 7 |
| Promotion features + `is_promo_tracked` | Section 8 |
| Holiday-specific past effect, `is_earthquake_period` | Section 9 |
| Oil (sirf lagged, test ke baad) | Section 10 (kamzor umeed) |
| Transactions: NAHI (same day), sirf 16+ din purani agar test mein faida de | Section 11 |
| Carnaval se pehle ka Sunday (naya event) | Section 14 |
| Store ki umar (days since opening) | Section 17 |
| Promotion relative to series ka normal level | Section 8, 15 |
| Local/regional holiday flags: kamzor, test karna | Section 16 |

## Phase 06 (model aur evaluation) ke faisle

- Target ko `log1p(sales)` par train karna (Section 1).
- Metrics: RMSLE aur WAPE, saath mein family, store aur demand-class ke hisaab se (Section 1, 6).
- Har model ko baseline (pichle hafton ki average) se behtar hona zaroori hai.
- Training data 1 June 2015 se (Section 2b); purana data sirf lag features ke liye.
- Ek global model saari series ke liye (Section 18). Store 52 ki accuracy alag se check (Section 17).

## Missing values ka faisla

- Sirf 3 columns mein missing values hain, aur teeno mein khaali hona khud information hai (Section 0b). Koi imputation nahi.

## Phase 03 (outliers) ke liye input

- Extreme values ki list aur un ke din (Section 14), jinhein wajah ke saath classify karna hai.
- Yaad rahe: Christmas, 2 January, holidays, promotions aur zalzala asli demand hain, ghaltiyan nahi.
