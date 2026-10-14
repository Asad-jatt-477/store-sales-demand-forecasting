# Phase 03 khulasa: Outliers Detection and Treatment

## Is run ke numbers
- Scope rows: 1,308,738 | robust-z extremes: 8,267 | IQR outliers: 12,572
- Final actions (test pass): koi nahi
- Badli gayi rows: 0 | units change: +0.000%
- Seed noise: 0.1031 WAPE points

*(Neeche ka narrative original notebook run se hai; naye data par numbers thore mukhtalif ho sakte hain.)*
## Detection (2 visual + 2 statistical)
1. **Boxplot:** har family ki range dikhata hai. Lekin upar wale "outliers" zyada tar do chhoti families (HOME APPLIANCES, BABY CARE), bare stores aur peak dinon ke hain, kyun ke 54 stores ek dabbe mein hain.
2. **Time-series + normal band:** dikhata hai ke har "bahar" wala din ek jaisa nahi (season, holiday, bulk order).
3. **IQR (har series):** zyada tar flags chhoti, aksar zero wali series ke season par hain, aur din ka context nahi dekhta.
4. **Contextual robust z:** har value ko usi series ke usi weekday ke normal se compare karta hai. **Primary detector.**

## Wajah ki tafteesh
Original run mein ~76% extremes (6,296 of 8,267) ke data mein nishaan mile: same family bohat se stores mein, company ka khaas din, store ka rush, promotion, aur calendar (Carnaval Sunday, Good Friday, Mother's Day, election ke din, store 1 ki holidays). **Yeh asli demand hain.**

## Treatment (data ki nature + forecast test, 4 windows, seed noise ke muqable)
| Wajah | Candidate rule | Test (original run) | Final |
|---|---|---|---|
| Nishaan wale extremes (H1-H4, L1-L3) | Rakho | - | **Rakho** |
| Intermittent spikes | Rakho | - | **Rakho** |
| Isolated spike (H5) | Cap | 4/4 windows mein thora kharab | **Rakho** |
| Isolated drop (L4) + store takreeban band (L0) | Normal se bharo | Faida noise se chhota | **Rakho** |

**Final nateeja (original run): data mein koi value nahi badli.** Yeh "kuch na karna" nahi, balkay test se sabit faisla hai ke is data aur is model setup ke liye outlier treatment ki zaroorat nahi. Validation/test nahi chhuye gaye, koi row delete nahi hui. Har candidate aur uska faisla `reports/03_outlier_treatment_log.csv` mein hai.

## Is phase ke sabaq
1. Jo rules kaghaz par theek lagte thay (bulk spikes kaatna, stock-out bharna), test mein fail hue. Treatment ka faisla **andaze se nahi, test se** hota hai.
2. Model comparison mein **noise naapna zaroori hai.** Seed test ke baghair 0.03-0.05 points ka farq "faida" lag raha tha, jab ke noise ~0.10 tha.

## Limitations (imaandari)
- Bulk order aur ghalat entry mein farq data se nahi ho sakta (receipt-level data nahi).
- Stock-out ka direct saboot (inventory data) nahi. L4 ko "stock-out" kehna sab se munasib wazahat hai, saboot nahi.
- Thresholds (20%, 5%, 1.3x, 0.5x, 2x, z = 5, Poisson 0.001) chune gaye hain. Sensitivity test sirf share thresholds ka hua.
- Rules poori training history dekh kar banaye gaye, is liye May/Dec 2016 ke test 100% andhe nahi (June 2017 window isi liye shaamil).
- Seed noise sirf 2 seeds se naapa gaya. Zyada seeds se andaza aur pakka hota.

## Phase 04 (Post-EDA) ke liye notes
- Patterns (weekday, payday, month) ko outlier labels ke sath dobara dekhna, misaal: "asli event" wale din hata kar ya alag kar ke.
- **Store x holiday** ka asar alag hai (store 1 mein holidays par customers ~70% kam).
- **Baar baar spike wali series** (H5, 5+ spikes) ek alag demand class ho sakti hain.
- Calendar din jo holidays file mein nahi: **election dry law**, **holiday se pehle ka Sunday**. Forecast window (16-31 Aug 2017) mein yeh nahi aate, lekin features ke taur par check karne layak hain.
