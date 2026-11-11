import json,random
from pathlib import Path
D=Path(__file__).parent
registry=json.loads((D/'training_registry.json').read_text());banned=set(registry['words'])
roots=[('FISH','BASS PERCH SOLE CARP PIKE SKATE RAY FLOUNDER'),('COMPUTING TERMS','MOUSE WINDOW CACHE COOKIE THREAD PROCESS TOKEN SERVER'),('GEMSTONES','RUBY OPAL EMERALD GARNET AMBER JADE ONYX SAPPHIRE'),('BASEBALL TERMS','BAT BALL DIAMOND BASE MITT STRIKE WALK RUN'),('PUNCTUATION','PERIOD COMMA COLON DASH HYPHEN BRACKET QUOTE SLASH'),('CHESS TERMS','KING ROOK PAWN BISHOP CHECK MATE FILE RANK'),('CLOTHING DETAILS','CAP SOCK GLOVE BELT CUFF COLLAR POCKET HEM'),('ART SUPPLIES','BRUSH PALETTE CANVAS EASEL CHARCOAL INK PENCIL PASTEL')]
for cat,words in roots:assert not set(words.split())&banned,(cat,set(words.split())&banned)
train=json.loads(Path('data/processed/train.json').read_text());vocab=sorted({w for p in train for w in p['words'] if w.isalpha()});rng=random.Random(225)
seeds=[rng.sample(vocab,4) for _ in roots]
(D/'fresh_seeds.json').write_text(json.dumps({'seed':225,'source':'Sorted distinct alphabetic input words in local data/processed/train.json; random.Random(225).sample four per family.','families':[{'id':f'new-{i+1:02d}','seed_words':s} for i,s in enumerate(seeds)]},indent=2)+'\n')
(D/'fresh_root_pools.json').write_text(json.dumps([{'id':f'new-{i+1:02d}','words':words.split()} for i,(cat,words) in enumerate(roots)],indent=2)+'\n')
raw=[
[('BODY POSTURES','body_postures','PERCH SQUAT CROUCH KNEEL RECLINE LOUNGE HUNCH SPRAWL','Positions or ways of positioning the human body.'),('VOICE DISSATISFACTION','express_discontent','CARP COMPLAIN GRUMBLE GROUSE WHINGE WHINE MOAN KVETCH','Verbs for voicing dissatisfaction, rather than attacking a target.'),('POLE WEAPONS','pole_weapons','PIKE SPEAR LANCE HALBERD GLAIVE PARTISAN TRIDENT JAVELIN','Weapons consisting of a long shaft with a pointed or bladed head.'),('ACT INEPTLY','inept_performance','FLOUNDER FUMBLE BUMBLE STUMBLE FALTER FLAIL BLUNDER BOTCH','Verbs describing struggling or performing clumsily.')],
[('MODEL ORGANISMS','model_organisms','MOUSE ZEBRAFISH NEMATODE YEAST BACTERIUM FLY FROG RABBIT','Organisms routinely used in biological research.'),('BAKED SWEETS','baked_sweets','COOKIE BROWNIE BLONDIE MACARON MERINGUE CUPCAKE TART ECLAIR','Sweet treats prepared by baking.'),('SENTIMENTAL OBJECTS','sentimental_objects','TOKEN MEMENTO KEEPSAKE SOUVENIR REMEMBRANCE HEIRLOOM RELIC MEMORABILIA','Objects kept as reminders of people, places, events, or affection.'),('RESTAURANT ROLES','restaurant_roles','SERVER CHEF BUSSER HOST SOMMELIER DISHWASHER BARTENDER CASHIER','Job roles found in restaurants.')],
[('PROGRAMMING LANGUAGES','programming_languages','RUBY PYTHON JAVA RUST SWIFT SCALA GO KOTLIN','Names of programming languages.'),('___ CITY','____city','EMERALD FORBIDDEN WINDY MAGIC ROSE CRESCENT HOLY ETERNAL','Words preceding CITY in established city nicknames.'),('REBECCA SUGAR CHARACTERS','steven_universe_characters','GARNET STEVEN CONNIE LARS SADIE ONION GREG JASPER','Exact single-word character names from Steven Universe.'),('___ BLUE','____blue','SAPPHIRE NAVY ROYAL BABY SKY POWDER MIDNIGHT STEEL','Words preceding BLUE to name familiar shades.')],
[('___MAN CHARACTERS','____man_characters','BAT SPIDER IRON SUPER SAND AQUA ANT ICE','Stems joining MAN in established comic-book character names, allowing normal hyphenation.'),('SOCIAL EVENTS','social_events','BALL GALA SOIREE RECEPTION BANQUET FESTIVAL FETE GATHERING','Organized social gatherings or celebrations.'),('ANNIVERSARY MATERIALS','anniversary_materials','DIAMOND PAPER WOOD TIN COPPER BRONZE CRYSTAL LINEN','Materials traditionally associated with wedding anniversaries.'),('ALKALINE TERMS','alkaline_terms','BASE LYE AMMONIA BORAX POTASH LIME HYDROXIDE ALKALI','Names or class terms for basic chemicals.')],
[('MOVE HURRIEDLY','move_hurriedly','DASH HURRY RUSH SCURRY HASTEN HUSTLE RACE SPRINT','Verbs for moving or proceeding quickly.'),('PH PRONOUNCED F','ph_pronounced_f','HYPHEN PHONE PHOTO PHASE PHRASE GRAPH ORPHAN TROPHY','Each contains PH pronounced as the F sound.'),('TOURNAMENT TERMS','tournament_structure','BRACKET SEED DRAW ROUND BYE MATCHUP QUALIFIER FINAL','Terms for arranging and progressing through a tournament.'),('REDUCE','reduce_quantity','SLASH CUT TRIM PRUNE PARE CURTAIL DECREASE DWINDLE','Verbs meaning make or become smaller in amount.')],
[('___ SIZE','____size','KING FULL HALF FUN POCKET BITE FAMILY ECONOMY','Words preceding SIZE in familiar size descriptions, allowing normal hyphenation.'),('DECEIVE','defraud','ROOK SWINDLE FLEECE DEFRAUD CON CHEAT DUPE HOODWINK','Verbs for deceiving or cheating someone.'),('LOAN SECURITY','loan_security','PAWN COLLATERAL GUARANTEE PLEDGE SURETY LIEN MORTGAGE BOND','Things or legal arrangements securing an obligation.'),('CHRISTIAN CLERGY','christian_clergy','BISHOP PRIEST DEACON VICAR RECTOR CURATE PASTOR CHAPLAIN','Titles or roles of Christian clergy.')],
[('___ DRAWER','____drawer','SOCK DESK CASH UTILITY KNIFE SILVERWARE KITCHEN UNDERWEAR','Words preceding DRAWER to describe a drawer by furniture, contents, or use.'),('___ COMPARTMENT','____compartment','GLOVE ENGINE LUGGAGE PASSENGER STORAGE BATTERY FREEZER SECRET','Words preceding COMPARTMENT in familiar expressions.'),('DOUBLED FINAL LETTERS','doubled_final_letters','CUFF JAZZ FUZZ BUZZ HISS MOSS BELL DOLL','Every word ends in a doubled consonant.'),('APPREHEND','apprehend','COLLAR ARREST NAB CATCH CAPTURE DETAIN BUST PINCH','Verbs for taking someone into custody or catching them.')],
[('WOVEN FABRICS','woven_fabrics','CANVAS DENIM TWEED SERGE CORDUROY BROADCLOTH GABARDINE BURLAP','Names of fabrics made by weaving.'),('BARBECUE FUELS','barbecue_fuels','CHARCOAL PROPANE BUTANE MESQUITE HICKORY OAK MAPLE CHERRY','Gases, charcoal, or woods used as barbecue fuel.'),('WRITE ON DOCUMENTS','document_signing','INK SIGN AUTOGRAPH ENDORSE INSCRIBE EXECUTE INITIAL COUNTERSIGN','Verbs for writing on documents, especially adding signatures or endorsements.'),('___ IN','____in','PENCIL TUNE FILL LOG CHECK WEIGH PLUG STEP','Words preceding IN in familiar phrasal verbs.')]
]
stories=[
['A fisherman gave the stubborn motor a KICK.','He called his accidental splash a GAFFE.','A passing DUDE helped him retrieve the tackle.','They nicknamed their boat GOLDILOCKS and tried a quieter fishing spot.'],
['An exhibit promised a REVOLUTION in household computing.','Its games could ENGROSS even skeptical visitors.','A family touring AMERICA stopped at the booth.','Their daughter GEN used a mouse to order a virtual cookie.'],
['A BANANA seller admired the gems in the next stall.','He watched ice begin to MELT around his fruit.','With ZERO customers waiting, he visited the jeweler.','The jeweler\'s KINKY curls framed a dazzling emerald pendant.'],
['A HONDA stopped beside a baseball field.','Its driver carried a sandwich filled with MUSSEL meat.','The players preferred MODERN equipment to the old wooden bats.','After practice they gathered at a nearby COTTAGE.'],
['A STALE newsletter needed fresh editing.','Its editor decided to SWITCH the punctuation style.','Her enthusiasm did not EBB as she adjusted every bracket.','By night she had BUILT a tidy guide to dashes and hyphens.'],
['The chess club jokingly called its president NEPO.','He said SORRY after knocking over a rook.','A curious cat began to ROAM across the board.','They put the pieces DOWN and offered it a cushion.'],
['A seamstress searched for the ROOT of a snag.','She made one careful PASS with her needle.','An ORIGAMI bird sat beside her sewing basket.','She finished a LAVISH collar and folded the spare gloves.'],
['A potter set aside her CLAY to visit an art shop.','The clerk recommended a MUSTARD pigment.','A tiny BUG crawled across a canvas sample.','She bought drawing supplies and left when the closing bell began its RINGS.']]
# Export only after checking every new input against all registered and fresh inputs.
scores=json.loads((D/'fresh_root_scores.json').read_text())['pools'];trees=[];used=set(banned);keys=set(registry['category_keys'])
for i,((cat,root),rs) in enumerate(zip(roots,raw)):
    tid=f'new-{i+1:02d}';anchors=scores[tid]['selections']['yellow']['words'];groups=[]
    for j,((label,key,pool,connection),anchor) in enumerate(zip(rs,anchors)):
        words=pool.split();assert len(words)==8 and len(set(words))==8,(tid,label)
        assert anchor in words,(tid,label,anchor)
        assert not set(words)&used,(tid,label,set(words)&used)
        assert not (set(words)&set(anchors)-{anchor}),(tid,label,'other anchor')
        assert key not in keys,(tid,key)
        used.update(words);keys.add(key)
        if '___' in label:
            style='fill_in_the_blank';fits={w:label.replace('___',w) for w in words}
        elif key in ['ph_pronounced_f','doubled_final_letters']:
            style='wordplay';fits={w:f'{w}: {connection}' for w in words}
        else:
            style='names_of_things';fits={w:f'{w}: {connection}' for w in words}
        groups.append({'id':f'{tid}-g{j}','category':label,'category_key':key,'style':style,'connection':connection,'words':words,'required':[anchor],'anchor':anchor,'anchor_alternate_meaning':fits[anchor],'word_fits':fits})
    altcat,altwords=roots[(i+3)%len(roots)]
    trees.append({'id':tid,'method':'false_connection','seed_words':seeds[i],'story':stories[i],'root_proposals':[{'category':cat,'words':root.split(),'connection':f'Each belongs to {cat.lower()}.'},{'category':altcat,'words':altwords.split(),'connection':f'Each belongs to {altcat.lower()}.'}],'root_category':cat,'root_pool':root.split(),'false_group':{'category':cat,'words':anchors},'groups':groups})
(D/'fresh_trees.json').write_text(json.dumps(trees,indent=2)+'\n');print('Wrote',len(trees),'families;',len(used)-len(banned),'globally new pool words')
