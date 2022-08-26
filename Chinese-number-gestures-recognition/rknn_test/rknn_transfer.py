from rknn.api import RKNN

INPUT_SIZE = 64

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # Config for Model Input PreProcess
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255,255,255]], reorder_channel='0 1 2', target_platform=['rk3399pro'])
    #rknn.config(channel_mean_value='0 0 0 255', reorder_channel='2 1 0')

    # Load TensorFlow Model
    print('--> Loading model')
    rknn.load_tensorflow(tf_pb='../digital_gesture_recognition/model_2500/digital_gesture.pb',
                         inputs=['input_x'],
                         outputs=['probability'],
                         input_size_list=[[INPUT_SIZE, INPUT_SIZE, 3]])
    print('done')

    # Build Model
    print('--> Building model')
    rknn.build(do_quantization=False, dataset='./dataset.txt')
    print('done')

    # Export RKNN Model
    rknn.export_rknn('./digital_gesture.rknn')

    # Release RKNN Context
    rknn.release()
