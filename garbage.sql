/*
 Navicat Premium Data Transfer

 Source Server         : MySQL
 Source Server Type    : MySQL
 Source Server Version : 90300
 Source Host           : localhost:3306
 Source Schema         : garbage

 Target Server Type    : MySQL
 Target Server Version : 90300
 File Encoding         : 65001

 Date: 25/05/2026 00:54:04
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for models
-- ----------------------------
DROP TABLE IF EXISTS `models`;
CREATE TABLE `models`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `path` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `arch` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `size_mb` double NULL DEFAULT NULL,
  `num_classes` int NULL DEFAULT NULL,
  `epoch` int NULL DEFAULT NULL,
  `val_acc` double NULL DEFAULT NULL,
  `created_at` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `updated_at` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `name`(`name` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 4 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of models
-- ----------------------------
INSERT INTO `models` VALUES (3, 'resnet50_20260422_161046.pth', 'models\\uploaded\\resnet50_20260422_161046.pth', 'resnet50', 90.3, 40, 10, 0.8687038280725319, '2026-05-14 23:26:29', '2026-05-14 23:26:29');

-- ----------------------------
-- Table structure for records
-- ----------------------------
DROP TABLE IF EXISTS `records`;
CREATE TABLE `records`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `thumb` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
  `label` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `confidence` double NULL DEFAULT NULL,
  `elapsed` double NULL DEFAULT NULL,
  `model` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `weight` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `user` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `created_at` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 10 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of records
-- ----------------------------
INSERT INTO `records` VALUES (1, 'https://picsum.photos/seed/rec1/64/64', '有害垃圾：过期药品/药片包装', 0.9956, 0.0489, 'ResNet', 'ResNet_model_92.40%.pth', 'admin', '2024-12-17 08:37');
INSERT INTO `records` VALUES (2, 'https://picsum.photos/seed/rec2/64/64', '可回收物：塑料瓶', 0.9863, 0.0534, 'ResNet', 'ResNet_model_92.40%.pth', 'admin', '2024-12-17 08:33');
INSERT INTO `records` VALUES (3, 'https://picsum.photos/seed/rec3/64/64', '厨余垃圾：果皮', 0.9919, 0.0398, 'ResNet', 'ResNet_model_92.40%.pth', 'admin', '2024-12-17 08:30');
INSERT INTO `records` VALUES (4, '/static/uploads/upload_caaaecaa9a5f4778b7d5b5a1fb1b7630.jpg', '有害垃圾：过期药品/药片包装', 0.9056, 0.048, 'ResNet', 'resnet50_20260422_161046.pth', 'admin', '2026-04-22 20:29');
INSERT INTO `records` VALUES (5, '/static/uploads/upload_70df2dd9e5b14d778ac9e58eb89aa51d.jpg', '有害垃圾：过期药品/药片包装', 0.9056, 0.048, 'ResNet', 'resnet50_20260422_161046.pth', 'admin', '2026-04-22 20:30');
INSERT INTO `records` VALUES (6, '/static/uploads/upload_aa44efefd3fb421dbb1fbb4c3f09dcb3.jpg', '有害垃圾：过期药品/药片包装', 0.9056, 0.048, 'ResNet', 'resnet50_20260422_161046.pth', 'admin', '2026-04-22 20:30');
INSERT INTO `records` VALUES (7, '/static/uploads/upload_28d141fcef124feebfa9bf7487924b3f.jpg', '有害垃圾：过期药品/药片包装', 0.9056, 0.048, 'ResNet', 'resnet50_20260422_161046.pth', 'admin', '2026-04-22 20:31');
INSERT INTO `records` VALUES (8, '/static/uploads/upload_c7ebb6fcd63448e7a8aff4b63a7aeb0c.jpg', '有害垃圾：过期药品/药片包装', 0.9056, 0.048, 'ResNet', 'resnet50_20260422_161046.pth', 'admin', '2026-04-22 20:32');
INSERT INTO `records` VALUES (9, 'D:\\project\\garbage-classifier-flask\\static\\uploads\\upload_77320a05f3d64ef5aba766f7fd70ac41.png', '有害垃圾：过期药品/药片包装', 0.9056, 0.048, 'ResNet', 'resnet50_20260422_161046.pth', 'admin', '2026-05-02 22:22');

-- ----------------------------
-- Table structure for settings
-- ----------------------------
DROP TABLE IF EXISTS `settings`;
CREATE TABLE `settings`  (
  `k` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `v` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
  PRIMARY KEY (`k`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of settings
-- ----------------------------
INSERT INTO `settings` VALUES ('current_model', 'resnet50_20260422_161046.pth');

-- ----------------------------
-- Table structure for users
-- ----------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `password` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `gender` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `email` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `phone` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `role` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `is_active` tinyint NOT NULL DEFAULT 1,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `username`(`username` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of users
-- ----------------------------
INSERT INTO `users` VALUES (1, '123', 'pbkdf2:sha256:1000000$BMgyvQe8BLIJIGdi$78e5792712a4807b3b62d4b328c7d72f7951c2ddd10bd2574a9ea9befa4e8658', '张三', '男', '123@qq.com', '1234567899', '普通用户', 1);
INSERT INTO `users` VALUES (2, 'admin', 'pbkdf2:sha256:1000000$BMgyvQe8BLIJIGdi$78e5792712a4807b3b62d4b328c7d72f7951c2ddd10bd2574a9ea9befa4e8658', '管理员', '男', 'admin@example.com', '1234567890', 'admin', 1);

SET FOREIGN_KEY_CHECKS = 1;
