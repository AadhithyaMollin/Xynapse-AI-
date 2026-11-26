# BIO-1/BIO-2 position-feature request

Predictions are computational hypotheses, not experimental findings.

Please provide position tables for the following sampled training proteins where feasible. IDs are kept as their dataset identifiers unless an accession is explicitly verified by exact sequence match. Include `protein_id`, 1-based `position`, and `wt` in every position table.

Requested position-table columns:

- `pos_structure`: protein_id, position, wt, rsasa, sasa_abs, ss8, ss3, n_contacts_8A, plddt
- `pos_evolution`: protein_id, position, wt, cons_entropy, cons_weighted, gap_frac, n_eff
- `pos_function`: protein_id, position, wt, is_active_site, is_binding_site, is_metal, is_dna_binding, is_disulfide, is_interface, dist_to_nearest_site
- Per-mutation evolution: protein_id, position, wt, mut, msa_logodds

Training protein requests (dataset ID followed by exact sequence):

## 1GB4.pdb_A27K (Tsuboyama, length 57)

`MTTFKLIINGKTLKGEITIEAVDAAEKEKIFKQYANDNGIDGEWTYDDATKTFTVTE`

## 2KZJ.pdb (Tsuboyama, length 50)

`KEMRNAYWEIALLPNLNNQQKRAFIRSLYDDPSQSANLLAEAKKLNDAQA`

## 2M5A.pdb (Tsuboyama, length 58)

`VDNKFNKEQQNAFYEILHLPNLNEEQRNAFIQSLKDDPSQSANLLAEAKKLNDAQAPK`

## 2ZW1.pdb_Y34A (Tsuboyama, length 57)

`MDTYKLILNGKTLKGETTTEAVDAATAEKVFKHAANEHGVHGHWTYDPETKTFTVTE`

## 5VWE.pdb (Tsuboyama, length 54)

`MSAYMLWLNASREKIKSDHPGISITDLSKKAGEIWKGMSKEKKEEWDRKAEDAR`

## EA|run1_0153_0005.pdb (Tsuboyama, length 47)

`IHIHVHDGDLEIHLHNTSEEILKLVEELLRRGLTVEEILKIVHEAFR`

## EEHEE_rd3_0809.pdb (Tsuboyama, length 43)

`GTVVVGNTKIKFPNPEKAEQFAKEIAKKLNLTVHTDGNTFTVK`

## EHEE_rd2_0251.pdb (Tsuboyama, length 40)

`TEEYTYTDKEKADRAAKEWANRGYTVTVTQDGPTYRVRVG`

## GG|run3_0687_0004.pdb (Tsuboyama, length 47)

`TDVTFHVNGRNLHLKGADEELLRRIEELIRRGADIEELEKAVKEWEK`

## HHH_rd2_0090.pdb (Tsuboyama, length 43)

`GTEDLKEEAKKALKKGDKERAKRALKKAGLSEDQIKKFIKEVL`

## HHH_rd4_0017.pdb (Tsuboyama, length 43)

`PEEIKREVEKLLRNGISEEEIVEILKRRGIDEEEVRRVIKKVS`

## NUD15_HUMAN_Suiter_2020 (ProteinGym, length 164)

`MTASAQPRGRRPGVGVGVVVTSCKHPRCVLLGKRKGSVGAGSFQLPGGHLEFGETWEECAQRETWEEAALHLKNVHFASVVNSFIEKENYHYVTILMKGEVDVTHDSEPKNVEPEKNESWEWVPWEELPPLDQLFWGLRCLKEQGYDPFKEDLNHLVGYKGNHL`

## PRKN_HUMAN_Clausen_2023 (ProteinGym, length 465)

`MIVFVRFNSSHGFPVEVDSDTSIFQLKEVVAKRQGVPADQLRVIFAGKELRNDWTVQNCDLDQQSIVHIVQRPWRKGQEMNATGGDDPRNAAGGCEREPQSLTRVDLSSSVLPGDSVGLAVILHTDSRKDSPPAGSPAGRSIYNSFYVYCKGPCQRVQPGKLRVQCSTCRQATLTLTQGPSCWDDVLIPNRMSGECQSPHCPGTSAEFFFKCGAHPTSDKETSVALHLIATNSRNITCITCTDVRSPVLVFQCNSRHVICLDCFHLYCVTRLNDRQFVHDPQLGYSLPCVAGCPNSLIKELHHFRILGEEQYNRYQQYGAEECVLQMGGVLCPRPGCGAGLLPEPDQRKVTCEGGNGLGCGFAFCRECKEAYHEGECSAVFEASGTTTQAYRVDERAAEQARWEAASKETIKKTTKPCPRCHVPVEKNGGCMHMKCPQPQCRLEWCWNCGCEWNRVCMGDHWFDV`

## r11_80_TrROS_Hall.pdb (Tsuboyama, length 49)

`KKVLVLSEDEIAELLGLSKEEIENMSEEEKEKLALELAKKLGIDEVIIT`

## r9_340_TrROS_Hall.pdb (Tsuboyama, length 69)

`KMKVIKTFENKETGEKIVIVKDEKTKKYYIFIYNKDGELVEKIEVEPGETITLPNGKTITIKEGDVIEF`

Please also provide a per-mutation `msa_logodds` table and review the PTEN `pos_evolution` table, whose saved entropy is degenerate at most positions. The PTEN/TP53 `pos_function.dist_to_nearest_site` field is null in the current teammate table; please fill it where supported by the annotation source, otherwise retain null and state why. No Tier-2 biology features were computed by this pipeline.
