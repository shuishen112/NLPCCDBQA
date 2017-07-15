'''
import ConfigParser
class Config(object):
	"""Holds model hyperparams and data information.
    
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    """General"""
    dataset = 'wiki'
    train_data = './data'+ dataset + '/train.txt'
    val_data='./data'+ dataset + '/dev.txt'
    test_data = './data'+ dataset + '/test.txt'
    isEnglish = True
    en_embedding_file = 'embedding/aquaint+wiki.txt.gz.ndim=50.bin'
    ch_embedding_file = 'embedding/'
    ch_stopwords = 'model/chStopWordsSimple.txt'

    # Model Hyperparameters
    embedding_dim = 50
    filter_sizes = "1,2,3,5"
    num_filters = 128
    dropout_keep_prob = 1
    l2_reg_lambda = 0.000001
    learning_rate = 1e-3
    loss = 'pair_wise'
    extend_feature_dim = 10

    # Training parameters
    batch_size = 64
    trainable = False
    num_epochs = 500
    evaluate_every = 500
   	checkpoint_every = 500
   	overlap_needed = True
   	dns = False #whether use dns samples
   	fresh = True #wheather recalculate the embedding or overlap default is True

   	# Misc Parameters
   	allow_soft_placement = True
   	log_device_placement = False

   	def saveConfig(self, filePath):
   		cfg = ConfigParser.ConfigParser()
   		cfg.add_section('General')
   		cfg.set('General', 'train_data', self.train_data)
   		cfg.set('General','val_data',self.val_data)
   		cfg.set('General','test_data',self.test_data)
   		cfg.set('General','isEnglish')
   		cfg.set('General','en_embedding_file',self.en_embedding_file)
   		cfg.set('General','ch_embedding_file',self.ch_embedding_file)
   		cfg.set('General','ch_stopwords',self.ch_stopwords)

   		cfg.set('General','embedding_dim',50)
   		cfg.set('General','filter_sizes','1,2,3,5')
   		cfg.set('General','num_filters',128)
   		cfg.set('General','dropout_keep_prob',1)
   		cfg.set('General','l2_reg_lambda',0.000001)
   		cfg.set('General','learning_rate',0.001)
   		cfg.set('General','loss','pair_wise')
   		cfg.set('General','extend_feature_dim',10)
'''
from tensorflow import flags
# Model Hyperparameters
flags.DEFINE_integer("embedding_dim",300, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
flags.DEFINE_float("learning_rate", 1e-3, "learn rate( default: 0.0)")
flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
flags.DEFINE_string("loss","pair_wise","loss function (default:point_wise)")
flags.DEFINE_integer('extend_feature_dim',10,'overlap_feature_dim')
# Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_boolean("trainable", False, "is embedding trainable? (default: False)")
flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
flags.DEFINE_boolean('overlap_needed',False,"is overlap used")
flags.DEFINE_boolean('position_needed',False,'is position embedding used')
flags.DEFINE_boolean('dns','False','whether use dns or not')
flags.DEFINE_string('data','wiki','data set')
flags.DEFINE_string('pooling','attentive','max pooling or attentive pooling')
flags.DEFINE_float('sample_train',1,'sampe my train data')
flags.DEFINE_boolean('fresh',True,'wheather recalculate the embedding or overlap default is True')
flags.DEFINE_boolean('clean',True,'whether we clean the data')
flags.DEFINE_string('conv','wide','wide conv or narrow')
flags.DEFINE_integer('gpu',0,'gpu number')
# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#data_help parameters
flags.DEFINE_boolean('isEnglish',True,'whether is data is english')
flags.DEFINE_string('en_embedding_file','embedding/aquaint+wiki.txt.gz.ndim=50.bin','english embedding')
flags.DEFINE_string('ch_embedding_file','embedding/','chinese embedding')
flags.DEFINE_string('ch_stopwords','model/chStopWordsSimple.txt','chinese stopwords')

