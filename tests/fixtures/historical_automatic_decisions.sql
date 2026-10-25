INSERT INTO images(filename, timestamp) VALUES
    ('20260101_130000_historical_confirmed.jpg', '20260101_130000'),
    ('20260101_130100_historical_missing.jpg', '20260101_130100'),
    ('20260101_130200_historical_uncertain.jpg', '20260101_130200');
INSERT INTO detections(detection_id, image_filename, od_class_name, od_confidence,
    score, bbox_x, bbox_y, bbox_w, bbox_h, bbox_quality, unknown_score,
    decision_state, decision_level, status) VALUES
    (301, '20260101_130000_historical_confirmed.jpg', 'bird', 0.9,
        0.9, 0.1, 0.1, 0.3, 0.3, 0.9, 0.1, 'confirmed', NULL, 'active'),
    (302, '20260101_130100_historical_missing.jpg', 'bird', 0.9,
        0.9, 0.1, 0.1, 0.3, 0.3, 0.9, 0.1, NULL, NULL, 'active'),
    (303, '20260101_130200_historical_uncertain.jpg', 'bird', 0.9,
        0.9, 0.1, 0.1, 0.3, 0.3, 0.9, 0.1, 'uncertain', NULL, 'active');
INSERT INTO classifications(detection_id, cls_class_name, cls_confidence, rank) VALUES
    (301, 'Parus_major', 0.9, 1),
    (302, 'Parus_major', 0.9, 1),
    (303, 'Parus_major', 0.9, 1);
