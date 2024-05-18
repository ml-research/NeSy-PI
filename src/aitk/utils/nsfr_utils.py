# Created by jing at 30.05.23
def get_prob(v_T, NSFR, args):
    """
    if args.dataset_type == 'kandinsky':
        predicted = NSFR.predict(v=v_T, predname='kp')
    elif args.dataset_type == 'clevr':
        if args.dataset == 'clevr-hans3':
            predicted = NSFR.predict_multi(
                v=v_T, prednames=['kp1', 'kp2', 'kp3'])
        if args.dataset == 'clevr-hans7':
            predicted = NSFR.predict_multi(
                v=v_T, prednames=['kp1', 'kp2', 'kp3', 'kp4', 'kp5', 'kp6', 'kp7'])
    """
    return NSFR.get_target_prediciton(v=v_T, prednames=['kp'], device=args.device)
    # return predicted
