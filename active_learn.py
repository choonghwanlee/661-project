import torch
import numpy as np
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import metrics
from evaluate_performance import evaluate_on_images
import generate_np as gnp

def get_bounding_box(frame_size):
    # get bounding box from mask
    proj_eye_center, proj_eye_radius = (frame_size[0]/2, frame_size[1]/3), (0.25*frame_size[0], 0.07*frame_size[1])
    y_start, y_end = int(proj_eye_center[1]-proj_eye_radius[1]), int(proj_eye_center[1]+proj_eye_radius[1])
    x_start, x_end = int(proj_eye_center[0]-proj_eye_radius[0]), int(proj_eye_center[0]+proj_eye_radius[0])
    bbox = [x_start, y_start, x_end, y_end]
    return bbox

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def predict_mask_w_point(image):
    model_type = "vit_t"
    sam_checkpoint = "./weights/mobile_sam.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    img = cv2.imread(image)
    height, width, _ = img.shape

    #this is for bounding box
    prompt = get_bounding_box([width, height])

    predictor = SamPredictor(mobile_sam)
    predictor.set_image(img)

    input_point = np.array([[(width/2), height/3]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    #for bounding box:
    # masks, scores, logits = predictor.predict(
    #     box=[[prompt]]
    # )

    print(masks.shape)

    for i, (mask, score, logits) in enumerate(zip(masks, scores, logits)):
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


"""
stuff for acquisition functions
"""
def imageEncoder(img, model, preprocess, device):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def generateScore(image1, image2, model, preprocess, device):
    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(test_img, model, preprocess, device)
    img2 = imageEncoder(data_img, model, preprocess, device)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

def model_sim(image1, image2):
  #image1 and image2 are paths to the image
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
  model.to(device)
  return round(generateScore(image1, image2, model, preprocess, device), 2)

def ssim(image1, image2):
  image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation = cv2.INTER_AREA)
  image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  return metrics.structural_similarity(image1_gray, image2_gray, full=True)[0]*100

def overall_similarity(image1, image2, importance=[(1/2), (1/2)]):
  if (importance[0] + importance[1]) != 1:
    print("weights must sum to 1")
    return

  img1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
  img2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED)

  h, w, _ = img1.shape
  proj_eye_center, proj_eye_radius = (w/2, h/3), (0.25*w, 0.1*h)
  y_start, y_end = int(proj_eye_center[1]-proj_eye_radius[1]), int(proj_eye_center[1]+proj_eye_radius[1])
  x_start, x_end = int(proj_eye_center[0]-proj_eye_radius[0]), int(proj_eye_center[0]+proj_eye_radius[0])

  crop_img1 = img1[y_start:y_end, x_start:x_end]
  crop_img2 = img2[y_start:y_end, x_start:x_end]

  model_similarity_score = model_sim(crop_img1, crop_img2)
  ssim_score = ssim(crop_img1, crop_img2)

#   print(model_similarity_score)
#   print(ssim_score)

  return model_similarity_score*importance[0] + ssim_score*importance[1]

"""
this should use uncertainty and similarity to choose best data points to label
compare all the images with low certainty with the images that have already been 
"""
def hybrid_selector():
   # uncertainty sampling -> if circularity is <0.7 or if the score is <0.7

   # find the images with similarity below x (idk yet)
    # for all labelled images, compare with the current image

   # return images who don't meed the threshold
   return

def active_learn():
   # for each image, generate mask from the model
   # collect all the results (mask + certainty score + image name)
   # uhhh let's pause on this real quick
   return


# steps:
# for each image
    # generate mask from the model
    # option 1: do all the uncertainty and similarity calculations here and store the values (image name + final value)
    # option 2: store all the results (mask + certainty score + image name)
# option 1: go through the final results and select the ones that need to be manually annotated
# option 2: run hybrid selector on all the results and select the ones that need to be manually annotated
# if there are no things that need to be annotated, yay!
# else, finetune and save the model
# evaluate with test. if accuracy is above 

def test(mode='iris'):
    # this will be a dictionary with key imagename and value dictionary with 2 np arrays
    image_and_mask = {}
    # load test images
    print("\nIMAGES")
    test_image_data = np.load('./train_data/labeled/images.npz')
    lst = test_image_data.files
    print(len(lst))
    for item in lst:
        arr = test_image_data[item]
        image_and_mask[item] = {'image': arr}

    if mode == 'iris':
        print("\nIRISES")
        test_image_data = np.load('./train_data/labeled/iris.npz')
        lst = test_image_data.files
        print(len(lst))
        for item in lst:
            name = item.replace("_iris", "")
            image_and_mask[name]["mask"] = test_image_data[item]

    elif mode == 'pupil':
        print("\nPUPILS")
        test_image_data = np.load('./train_data/labeled/pupil.npz')
        lst = test_image_data.files
        print(len(lst))
        for item in lst:
            name = item.replace("_pupil", "")
            image_and_mask[name]["mask"] = test_image_data[item]

    images = []
    masks = []
    for im in image_and_mask:
        if "mask" in image_and_mask[im]:
            m_grayscale = np.asarray(Image.fromarray(np.array(image_and_mask[im]["mask"])).convert('L'))
            images.append(np.array(image_and_mask[im]["image"]))
            masks.append(m_grayscale)
    
    print()
    print(len(images))
    print(len(masks))
    mious = evaluate_on_images(images, masks)
    return mious


"""
print(test())
    #this is the mious from running test on the base model for iris (line above)
mious_base_iris = [0.3871262856324171, 0.3865523945869084, 0.4906237943672839, 0.49006462191358025, 0.4052900752314815, 0.4920454764660494, 0.49165629822530865, 0.3705327275763898, 0.49263768325617285, 0.40442925347222225, 0.492842399691358, 0.3971580825617284, 0.4896453028549383, 0.48857108410493827, 0.4926408179012346, 0.49225646219135805, 0.4899045138888889, 0.4894171971450617, 0.48535180362654323, 0.4917703510802469, 0.41138768325617286, 0.3901633936176267, 0.49155671296296294, 0.49310233410493826, 0.48854407793209875, 0.3645291469190874, 0.4698051697530864, 0.4026535976080247, 0.3904070346451354, 0.4918916377314815, 0.4914875096450617, 0.30901927821703185, 0.4767194733796296, 0.49288387345679013, 0.4906131847993827, 0.4904513888888889, 0.39215231271560147, 0.3910473188889194, 0.3981786730118374, 0.4922733410493827, 0.4928633777006173, 0.377425394990478, 0.40455078125, 0.4891632908950617, 0.4075428759369579, 0.4926466049382716, 0.492626350308642, 0.3906001170988424, 0.4925371334876543, 0.40764298804012344, 0.41057798032407405, 0.48715760030864197, 0.49251326195987655, 0.4925882523148148, 0.49023992091049384, 0.4899146412037037, 0.3812664239247185, 0.39531216540271286, 0.35175291458165114, 0.24248370527126753, 0.46536241319444444, 0.3941263843711043, 0.4923488136574074, 0.49224729938271605, 0.49168113425925924, 0.40682751453393745, 0.40147882908950616, 0.4919902584876543, 0.4920924961419753, 0.49004822530864195, 0.4903778452932099, 0.3878743256035673, 0.39745816168960124, 0.3399852585599098, 0.39626953125, 0.48882691936728395, 0.34979628615860126, 0.4925778838734568, 0.49092977082817923, 0.48960792824074073, 0.4899901138117284, 0.38477252020033004, 0.39463662229938273, 0.4919777199074074, 0.4895529513888889, 0.49288194444444444, 0.4009631141413956, 0.4887897858796296, 0.49253303433641976, 0.40425907457383214, 0.4925366512345679, 0.49254267939814816, 0.48866174768518517, 0.40757667824074073, 0.4886441454475309, 0.40918137538580246, 0.3772706873588511, 0.49225260416666666, 0.4928462577160494, 0.4900853587962963, 0.3049873715002211, 0.3264205671450201, 0.39168492844149255, 0.39679277584876543, 0.4917025945216049, 0.49102478780864195, 0.49088710455246914, 0.38904714407535024, 0.3814702177764126, 0.4020556037808642, 0.49154682677469136, 0.48961998456790123, 0.4778727816358025, 0.49222366898148145, 0.34565907437466953, 0.4885737364969136, 0.407543420966968, 0.4928614486882716, 0.4929542824074074, 0.39663194444444444, 0.49159095293209876, 0.49226128472222225, 0.49198037229938274, 0.4917947048611111, 0.4813524787808642, 0.40138273698670707, 0.40984780567506063, 0.3903419906926675, 0.3823897520732904, 0.49074387538580244, 0.4923939043209877, 0.47908058449074076, 0.4687133487654321, 0.35832511513943816, 0.43101393711419755, 0.3116010045732166, 0.38855933736029225, 0.4916775173611111, 0.39422293233844186, 0.4012208236882716, 0.4899587673611111, 0.40404369212962965, 0.48777006172839504, 0.4913544077932099, 0.49261091820987657, 0.36580714093539596, 0.36858870967741936, 0.49295452353395064, 0.41015263310185185, 0.40274570794753084, 0.49207585841049384, 0.49261405285493826, 0.40032542469659016, 0.3970739293981482, 0.49022882908950616, 0.4909174864969136, 0.4888428337191358, 0.49283347800925925, 0.3853649818481068, 0.3955925273631457, 0.4917616705246914, 0.4924040316358025, 0.4922993827160494, 0.3968569155092593, 0.40644599076291993, 0.4920247395833333, 0.4921624228395062, 0.4897535686728395, 0.38034781492218595, 0.35566520015998254, 0.4361140046296296, 0.49180362654320986, 0.48335358796296296, 0.3500312662416137, 0.36684635975492624, 0.4916823398919753, 0.4923982445987654, 0.4905167341820988, 0.38168130123920607, 0.40433400848765433, 0.4927510127314815, 0.492681568287037, 0.3751950875489445, 0.39759358907708675, 0.39600652598094677, 0.48857036072530863, 0.4906442901234568, 0.4062068383487654, 0.4077787422839506, 0.4053616898148148, 0.49237630208333333, 0.49254002700617283, 0.26640616612932766, 0.33257891570382875, 0.49348934220679014, 0.47584080825617286, 0.4927789834104938, 0.4900491898148148, 0.39509886188271603, 0.39878464703595856, 0.4919974922839506, 0.48943190586419755, 0.4046190200617284, 0.3933316350997857, 0.4908815586419753, 0.49158275462962964, 0.49310329861111113, 0.4039809625220752, 0.40648616872034443, 0.4922345196759259, 0.4918494405864198, 0.3874856464648949, 0.3891899229983897, 0.49305531442901235, 0.48870780285493826, 0.48755690586419753, 0.40882426697530866, 0.4062835165895062, 0.41095052083333333, 0.4924537037037037, 0.49292582947530866, 0.4896561535493827, 0.39462533812572853, 0.3529030485970524, 0.2463059623311462, 0.3905164930555556, 0.4873895640432099, 0.48098234953703706, 0.38265156404473744, 0.39962904609579697, 0.49199049961419755, 0.48621407214506174, 0.49342255015432096, 0.3873170894046299, 0.3135304184547015, 0.40511670524691357, 0.4071479552469136, 0.492416087962963, 0.49265721450617284, 0.396440731095679, 0.39282817322530866, 0.49009620949074073, 0.49119526427469135, 0.49255931712962964, 0.49228949652777776, 0.49022786458333334, 0.49173731674382715, 0.48901017554012344, 0.4086938175154321, 0.3873858013332684, 0.3906037727659636, 0.4921879822530864, 0.49188368055555554, 0.4927425733024691, 0.4890596064814815, 0.4720613908179012, 0.45584514853395064, 0.3597280362050835, 0.3849604296744392, 0.492310474537037]
print(len(mious_base_iris))
np_mious_iris = np.array(mious_base_iris)
print(np.average(np_mious_iris))
    # base miou average is 0.4436347319402451
"""

"""
print(test(mode='pupil'))
    #this is the mious from running test on the base model for pupil (line above)
mious_base_pupil = [0.39117250738851633, 0.38877446801938037, 0.4961566840277778, 0.4961653645833333, 0.4088773148148148, 0.4980859375, 0.49786241319444446, 0.37390745854820645, 0.4985279224537037, 0.40794560185185186, 0.4985710841049383, 0.40081404320987657, 0.49532190393518516, 0.49359881365740743, 0.4986889949845679, 0.4982441165123457, 0.49528211805555555, 0.49124156057098767, 0.497680362654321, 0.4146385513117284, 0.39357651975841473, 0.49701630015432097, 0.4982855902777778, 0.4945413773148148, 0.36635849849696234, 0.4732764274691358, 0.4069268422067901, 0.3946021660009134, 0.4977416087962963, 0.49785011574074073, 0.30885702363669554, 0.4826584201388889, 0.4988259548611111, 0.49675178433641975, 0.49644386574074073, 0.39502339040270074, 0.4023297646604938, 0.49820746527777776, 0.49866030092592595, 0.37723298134816596, 0.4085747010030864, 0.49476200810185184, 0.4110286458333333, 0.4982368827160494, 0.4979468074845679, 0.3945004258487798, 0.4981397087191358, 0.4111207561728395, 0.41407576195987655, 0.49836299189814814, 0.49879701967592593, 0.49657262731481483, 0.4954325810185185, 0.385109394128537, 0.3993800822859502, 0.3530282881573962, 0.2409824066035832, 0.47141203703703705, 0.3982424286265432, 0.49766541280864196, 0.49767361111111114, 0.4975952449845679, 0.410283589611279, 0.40473355516975307, 0.49778452932098766, 0.4979272762345679, 0.4960496238425926, 0.4959334008487654, 0.3917727725009562, 0.40164685480102374, 0.33959105079064417, 0.39959056712962965, 0.4946585648148148, 0.35071465553381104, 0.4983885513117284, 0.4970334201388889, 0.49577039930555555, 0.4958960262345679, 0.3987490354938272, 0.49785734953703703, 0.4987622974537037, 0.40437459550749283, 0.4943494405864198, 0.497905574845679, 0.40774281442901233, 0.49835624035493825, 0.49845775462962966, 0.49440007716049383, 0.41113642939814815, 0.49348644868827163, 0.41240644290123457, 0.38096476344533703, 0.4977370273919753, 0.49814019097222223, 0.49608772183641975, 0.3045434886284868, 0.3263674946731017, 0.396024787808642, 0.40118296682098764, 0.4980567611882716, 0.4964899209104938, 0.4970363136574074, 0.3931615543539697, 0.3848406789145279, 0.40627314814814813, 0.4973165027006173, 0.49566864390432097, 0.4838242669753086, 0.4981423611111111, 0.3462913878508734, 0.49369237075617284, 0.41111665702160494, 0.4982038483796296, 0.498365162037037, 0.40058232060185184, 0.4971141975308642, 0.4981030574845679, 0.49795524691358023, 0.49764708719135803, 0.4050067042231783, 0.4132785976080247, 0.3943688326820289, 0.3863092706141988, 0.49596764081790123, 0.49785300925925924, 0.4849679301697531, 0.4747485050154321, 0.3596140500566083, 0.43435305748456793, 0.31102564436790897, 0.3929511830253156, 0.498306568287037, 0.49782624421296295, 0.3983530977676397, 0.40503833912037035, 0.4961060474537037, 0.4074992766203704, 0.493778694058642, 0.49738450038580245, 0.498529369212963, 0.3686866273476872, 0.37150442853600196, 0.4986762152777778, 0.4134613715277778, 0.40606047453703703, 0.4979407793209877, 0.49823109567901236, 0.40413628472222224, 0.4011021894290123, 0.49584056712962965, 0.49623649691358024, 0.493833912037037, 0.49880714699074075, 0.38922985876853416, 0.3995777874228395, 0.49748963155864195, 0.4977131558641975, 0.4980473572530864, 0.4006057098765432, 0.41000337577160495, 0.4982749807098765, 0.4985177951388889, 0.49519941165123454, 0.38405832752122504, 0.35590221874201927, 0.4394401041666667, 0.497861930941358, 0.4896556712962963, 0.3511049470340006, 0.37020020601993137, 0.4975323109567901, 0.49816237461419755, 0.4965063175154321, 0.4083029513888889, 0.49867500964506173, 0.4987823109567901, 0.37370635307343614, 0.4015164448302469, 0.39966868893411395, 0.49410035686728393, 0.4957392939814815, 0.40972198109567903, 0.4109888599537037, 0.40887273341049385, 0.49772979359567904, 0.49847342785493826, 0.26567386459040815, 0.33315005684227406, 0.49935016396604937, 0.48218581211419753, 0.49895592206790124, 0.49610074266975307, 0.39941960841049384, 0.4028238065108527, 0.49786506558641974, 0.40864221643518517, 0.39616736790007206, 0.49613088348765433, 0.49712914737654323, 0.4985568576388889, 0.4074305555555556, 0.409800106095679, 0.4976810860339506, 0.4975882523148148, 0.3915505917517888, 0.39304277584876546, 0.4984326774691358, 0.49338951581790125, 0.4122774402006173, 0.40975453317901234, 0.41421489197530864, 0.4980046778549383, 0.49926432291666667, 0.49565996334876544, 0.3987478298611111, 0.35369214589132236, 0.24532060190662355, 0.3939296392746914, 0.4933347800925926, 0.48687909915123456, 0.40379650018277446, 0.4974283854166667, 0.4919965277777778, 0.49948398919753084, 0.3906489566678965, 0.3138367652604352, 0.40843629436728396, 0.41052107445987657, 0.49827570408950617, 0.49833839699074073, 0.40007643711419755, 0.396401668595679, 0.4955075713734568, 0.4967826485339506, 0.49867091049382717, 0.4984541377314815, 0.49605348186728393, 0.4976825327932099, 0.49403091242283953, 0.4121800250771605, 0.39129140919061683, 0.3941309405310308, 0.49743730709876544, 0.4975421971450617, 0.4980666473765432, 0.4949001736111111, 0.47800660686728397, 0.4591157889660494, 0.3617298714284749, 0.3889815450743136, 0.4985691550925926, 0.4972776813271605]
print(len(mious_base_pupil))
np_mious_pupil = np.array(mious_base_pupil)
print(np.average(np_mious_pupil))
    # base miou average is 0.4484484039157212
"""