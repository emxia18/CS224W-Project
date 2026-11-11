import json,random,re
from pathlib import Path
D=Path(__file__).parent
spec=json.loads((D/'extra_root_specs.json').read_text()); scores=json.loads((D/'extra_root_scores.json').read_text())['pools']
raw={
2:[('DELIGHTFUL PEOPLE','delightful_people','PEACH GEM JEWEL TREASURE ANGEL DARLING SWEETHEART DEAR','Affectionate nouns for a delightful or valued person.'),('___ JOB','____job','PLUM DREAM DAY NOSE SUMMER ODD DIRTY INSIDE','Each precedes JOB in a familiar expression.'),('FAILED PRODUCTS','failed_products','LEMON DUD CLUNKER TURKEY FLOP FAILURE WASHOUT BOMB','Nouns for unsuccessful or defective products or productions.'),('___FRUIT','____fruit','GRAPE PASSION DRAGON JACK BREAD STAR UGLI KIWI','Each forms the name of a fruit when followed by FRUIT; ordinary spaces are allowed.')],
3:[('LEADERSHIP TITLES','leadership_titles','CHAIR PRESIDENT SECRETARY TREASURER DIRECTOR GOVERNOR MANAGER LEADER','Titles of people with organizational leadership responsibilities.'),('PUT INTO WORDS','put_into_words','COUCH FORMULATE DESCRIBE WORD STATE ARTICULATE EXPRESS FRAME','Verbs meaning formulate or express something in words.'),('BODY SPECIMENS','body_specimens','STOOL URINE BLOOD SALIVA SPUTUM SEMEN TISSUE SWEAT','Materials collected from the body for medical testing.'),('GYM EQUIPMENT','gym_equipment','BENCH TREADMILL DUMBBELL BARBELL KETTLEBELL ELLIPTICAL ROWER TRAMPOLINE','Items of exercise equipment.')],
6:[('EVALUATE','evaluate','GRADE EVALUATE APPRAISE RATE ASSESS SCORE REVIEW GAUGE','Verbs meaning judge quality, value, or performance.'),('MEDICAL PROCEDURES','medical_procedures','EXAM BIOPSY X-RAY MRI ULTRASOUND ENDOSCOPY EKG ANGIOGRAM','Procedures or examinations used in medical diagnosis.'),('___ TUBE','____tube','TEST VACUUM CATHODE INNER FEEDING BREATHING FALLOPIAN NEON','Each precedes TUBE in a familiar technical or everyday expression.'),('QUESTION CLOSELY','question_closely','QUIZ QUESTION INTERROGATE GRILL PROBE DEBRIEF INTERVIEW QUERY','Verbs meaning seek information by questioning.')],
9:[('OUTER CASINGS','outer_casings','HULL SHELL HUSK RIND PEEL POD CASING CARAPACE','Protective or enclosing outer coverings.'),('MOVE EFFORTLESSLY','move_effortlessly','SAIL GLIDE COAST CRUISE DRIFT FLOAT SOAR WAFT','Verbs for moving smoothly or without apparent effort.'),('BROADCAST PRESENTERS','broadcast_presenters','ANCHOR EMCEE ANNOUNCER PRESENTER MODERATOR NEWSCASTER SPORTSCASTER INTERVIEWER','People who present or lead broadcast programs.'),('___ OVER','____over','KEEL TIP KNOCK BEND MULL GLOSS SKIM CARRY','Each precedes OVER in a familiar phrasal verb.')],
10:[('COMIC SIDEKICKS','comic_sidekicks','ROBIN SPEEDY BUCKY SUPERBOY AQUALAD BATGIRL SUPERGIRL JUBILEE','Names of comic book characters who have served as superhero sidekicks; compound hero names are conventionally also spaced.'),('END IN NCH','end_in_nch','FINCH CINCH WINCH LYNCH HAUNCH LAUNCH BLANCH TRENCH','Each word ends with the three letters NCH.'),('PLAYFUL ADVENTURES','playful_adventures','LARK PRANK ESCAPADE ANTIC CAPER FROLIC ROMP ADVENTURE','Nouns for playful or mischievous adventures and activities.'),('URGE INTO ACTION','urge_into_action','GOOSE PROD SPUR NUDGE GOAD EGG ENCOURAGE INCITE','Verbs meaning urge or stimulate someone into action.')],
17:[('GUIDE','guide','PILOT STEER GUIDE SHEPHERD CONDUCT ESCORT USHER NAVIGATE','Verbs meaning guide a person, group, or vehicle.'),('CUSTOMIZE','customize','TAILOR ADAPT CUSTOMIZE ADJUST MODIFY PERSONALIZE CONFIGURE CALIBRATE','Verbs meaning alter something to meet particular needs or settings.'),('FRUIT DESSERTS','fruit_desserts','COBBLER CRUMBLE BUCKLE BETTY CRISP PANDOWDY SLUMP FOOL','Names of fruit desserts.'),('CARETAKER ROLES','caretaker_roles','STEWARD CUSTODIAN GUARDIAN CARETAKER CURATOR WARDEN KEEPER CONSERVATOR','Roles responsible for caring for or safeguarding people or things.')]
}
train=json.loads(Path('data/processed/train.json').read_text());vocab=sorted({w for p in train for w in p['words'] if w.isalpha()});rng=random.Random(226)
allseeds=[rng.sample(vocab,4) for _ in range(20)]
banned=set(json.loads((D/'training_registry.json').read_text())['words'])
for t in json.loads((D/'fresh_trees.json').read_text()):
 for g in t['groups']:banned.update(g['words'])
banned.update('MEMENTO KEEPSAKE SOUVENIR REMEMBRANCE HEIRLOOM RELIC MEMORABILIA'.split())
out=[];seen=set()
for n,groups in raw.items():
 ident=f'extra-{n:02d}';cat,rs=spec[n-1];roots=rs.split();anchors=scores[ident]['selections']['yellow']['words'];assert anchors==[g[2].split()[0] for g in groups]
 seed=allseeds[n-1]
 t={'id':ident,'method':'false_connection','seed_words':seed,'story':[f'A traveler wrote {seed[0]} in a notebook before considering {cat.lower()}.',f'A sign reading {seed[1]} prompted a conversation about those objects.',f'A friend offered {seed[2]} as a clue to their setting.',f'They remembered {seed[3]} and compared the ordinary and alternate meanings of the words.'],'root_proposals':[{'category':cat,'words':roots,'connection':f'Each is a member or term of {cat.lower()}.'},{'category':spec[n%20][0],'words':spec[n%20][1].split(),'connection':'An alternative concrete theme considered from the story.'}],'root_category':cat,'root_pool':roots,'false_group':{'category':cat,'words':anchors},'groups':[]}
 for j,(label,key,ws,conn) in enumerate(groups):
  words=ws.split();assert len(words)==8
  collisions=set(words)&(banned|seen)
  if collisions:print(ident,label,'COLLISIONS',collisions)
  seen.update(words)
  t['groups'].append({'id':f'{ident}-g{j}','category':label,'category_key':key,'style':'fill_in_the_blank' if '___' in label else 'wordplay' if label=='END IN NCH' else 'names_of_things','connection':conn,'words':words,'required':[words[0]],'anchor':words[0],'anchor_alternate_meaning':f'{words[0]}: {conn}','word_fits':{w:f'{w}: {conn}' for w in words}})
 out.append(t)
(D/'extra_trees.json').write_text(json.dumps(out,indent=2)+'\n')
